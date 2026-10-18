/**
 * 带进度的模型下载。
 *
 * 抽成公共模块的原因与 ortRuntime.ts 一样：现在有两个消费者 ——
 * `segStore`（MobileSAM，42MB）与 `vpStore`（YOLOE，20MB），各写一份必然漂移。
 *
 * `stallMs`：**两个分片之间**超过这么久没有任何字节就判定卡死并中止。
 * 为什么用「停顿」而不是「总时长上限」：几十 MB 在慢网络上要几分钟是正常的，
 * 设总时长上限会把正常用户一起误伤；而**长时间零字节**在任何网络下都只可能是真卡住了。
 * 没有这个兜底，一次卡死会让状态永远停在 loading，而调用方在 loading 时
 * 通常会**静默吞掉所有后续点击** —— 按钮看起来就是彻底坏了，只能刷新页面。
 */
export async function fetchWithProgress(
  url: string,
  onPct: (pct: number) => void,
  stallMs = 20000,
): Promise<ArrayBuffer> {
  const ac = new AbortController();
  let stallTimer: number | undefined;
  const armStall = () => {
    if (stallTimer !== undefined) clearTimeout(stallTimer);
    stallTimer = window.setTimeout(() => ac.abort(), stallMs);
  };
  const disarmStall = () => {
    if (stallTimer !== undefined) {
      clearTimeout(stallTimer);
      stallTimer = undefined;
    }
  };

  try {
    armStall();
    const res = await fetch(url, { signal: ac.signal });
    if (!res.ok) {
      throw new Error(
        `模型文件下载失败（HTTP ${res.status}）。请确认 ${url} 存在 —— ` +
          '部署时 public/models/ 下的模型必须一并上传。',
      );
    }
    const total = Number(res.headers.get('Content-Length') ?? 0);
    if (!res.body || !total) {
      // 拿不到总长度（比如被分块传输）：进度条退化成不确定态，但下载照常
      onPct(-1);
      return await res.arrayBuffer();
    }
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      armStall();
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onPct(Math.min(100, Math.round((received / total) * 100)));
    }
    const merged = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    return merged.buffer;
  } catch (e) {
    if (ac.signal.aborted) {
      throw new Error(
        `模型下载中断：超过 ${Math.round(stallMs / 1000)} 秒没有收到任何数据，请检查网络后重试。`,
      );
    }
    throw e;
  } finally {
    disarmStall();
  }
}
