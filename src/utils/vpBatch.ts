/**
 * 「照样子找」批量的**纯逻辑**：每张图的统计、异常判定、以及给用户看的原因。
 *
 * 抽成独立模块（不 import store / React / onnx）是为了能直接被 `test:parse` 断言 ——
 * 批量的价值一半在「跑得完」，另一半在「跑完之后用户知道该看哪几张」，
 * 而后者全是阈值判断，写错就是静默的难用（要么一张都不标，要么标满误报），必须测。
 *
 * ## 为什么需要「异常标记」
 * 批量最大的风险不是慢，是**静默出错**：100 张图跑完，用户看到的是一句「完成」，
 * 于是默认它是好的。但实测里同一批图 F1 能从 0.000 到 0.750，
 * 「一个都没找到」和「找到一堆误报」在「完成」这句话里长得一模一样。
 * 所以要把可疑的张数单独点出来，并且**说清为什么可疑**。
 *
 * ## 阈值为什么是「相对中位数」而不是绝对值
 * 不同图的合理框数差很多（一张图 2 个怪，另一张 12 个），绝对阈值必然误报；
 * 而分数本身也会随阈值设置整体漂移（调低阈值就会涌进一批低分框），
 * 所以「分数偏低」也只能跟本批的中位数比。代价是**小批次（<3 张有框）不标** ——
 * 中位数在 2 个样本上没有意义，硬标只会制造狼来了。
 */

/** 一张图的批量结果统计 */
export interface VpBatchStat {
  imageId: string;
  /** 这张图上留下的候选框数（已过去重 / 越界 / 退化框） */
  count: number;
  /** 这些候选框的平均分；没有框时 0 */
  meanScore: number;
  /** 图片解码失败或推理抛错。与「没找到」是两件事，必须分开计数 */
  failed: boolean;
}

export type VpFlagKind = 'none' | 'many' | 'lowconf';

export interface VpBatchFlag {
  imageId: string;
  kind: VpFlagKind;
  /** 给用户看的一句话原因（含具体数字） */
  reason: string;
}

export interface VpBatchSummary {
  /** 有候选框的张数 */
  withBoxes: number;
  /** 候选框总数 */
  total: number;
  /** 一张都没找到的张数（不含失败） */
  empty: number;
  /** 解码 / 推理失败的张数 */
  failed: number;
  /** 框数中位数（**含**一张都没找到的图）；样本不足时为 0 */
  median: number;
  /** 平均分中位数；样本不足时为 0 */
  medianScore: number;
  /** 需要用户优先确认的张（含原因） */
  flags: VpBatchFlag[];
}

/** 至少要有几张「有框」的图才够格算中位数。低于这个数就不标，避免狼来了 */
export const MIN_STATS_SAMPLE = 3;

/** 中位数。空数组返回 0 —— 调用方必须先判断长度，这里不抛异常 */
export function medianOf(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  const s = xs.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * 「框特别多」的判定线。
 *
 * 取 `max(4, 中位数*3)`：3 倍是「这张图明显不正常」的经验值，
 * 而下限 4 是为了中位数为 1~2 时（稀疏目标）不至于把 3 个框也当成异常 ——
 * 那种图里多找出一个常常是真的。
 */
export function manyThreshold(median: number): number {
  return Math.max(4, Math.ceil(median * 3));
}

/**
 * 「分数普遍偏低」的判定线：本批平均分中位数的 70%。
 * 相对比较所以与量纲无关 —— 换模型、换阈值都不用重新标定这条线。
 */
export function lowConfThreshold(medianScore: number): number {
  return medianScore * 0.7;
}

/**
 * 由逐图统计算出汇总与异常标记。
 *
 * 三个判定互相独立、可以同时命中（一张图可以既多又分低），
 * 但 UI 上只显示**最值得看的那一条**，所以这里按 many → lowconf → none 的优先级
 * 每个 imageId 只出一条。
 */
export function summarizeBatch(stats: readonly VpBatchStat[]): VpBatchSummary {
  const ok = stats.filter((s) => !s.failed);
  const withBoxes = ok.filter((s) => s.count > 0);
  // 框数的中位数取**所有没失败的图**（含一张都没找到的），不是只取有框的。
  // 只取有框的话中位数恒 ≥1，「整批基本全空」就永远看不到了 ——
  // 而那恰恰是最该被喊停的情况（用户在对着一个不认示例的模型白跑 100 张）。
  const median = medianOf(ok.map((s) => s.count));
  // 平均分的中位数仍只取有框的图：没找到的图没有「平均分」可言，填 0 会把中位数拖垮
  const medianScore = medianOf(withBoxes.map((s) => s.meanScore));
  // 统计标记需要足够的样本，否则中位数没有意义（2 个样本的「中位数」只是运气）。
  //
  // 但有一条例外：**整批基本全是空的**时，「唯独某张图找出一堆」本身就是最强的异常信号，
  // 这时候根本不需要中位数 —— 绝对线（manyThreshold(0) = 4 个）就够用，
  // 而漏掉它恰恰是最危险的情况：用户会以为整批都失败了，其实是一张图在狂吐误报。
  const enough = withBoxes.length >= MIN_STATS_SAMPLE || (median === 0 && ok.length >= 3);

  const flags: VpBatchFlag[] = [];
  if (enough) {
    const hi = manyThreshold(median);
    const lo = lowConfThreshold(medianScore);
    for (const s of ok) {
      if (s.count === 0) {
        // 「没找到」不算异常判定，是独立的一类 —— 它可能是真的没有，
        // 所以文案必须把两种可能都写上，不能让用户以为程序坏了。
        continue;
      }
      if (s.count > hi) {
        flags.push({
          imageId: s.imageId,
          kind: 'many',
          reason:
            median > 0
              ? `找到 ${s.count} 个，是其他图的 ${Math.round(s.count / median)} 倍 —— 多半是误报，建议重点看`
              : `找到 ${s.count} 个，而其他图基本一个都没有 —— 多半是误报，建议重点看`,
        });
        continue;
      }
      if (medianScore > 0 && s.meanScore < lo) {
        const pct = Math.round((s.meanScore / medianScore) * 100);
        flags.push({
          imageId: s.imageId,
          kind: 'lowconf',
          reason: `找到 ${s.count} 个，但平均分只有其他图的 ${pct}% —— 多半是勉强凑数的，建议重点看`,
        });
      }
    }
  }

  return {
    withBoxes: withBoxes.length,
    total: withBoxes.reduce((n, s) => n + s.count, 0),
    empty: ok.filter((s) => s.count === 0).length,
    failed: stats.length - ok.length,
    median,
    medianScore,
    flags,
  };
}

/** 一句话总结批量结果，给 toast / 面板标题用 */
export function batchHeadline(s: VpBatchSummary): string {
  if (s.withBoxes === 0 && s.failed === 0) return '这批图里没找到像它的东西';
  const parts: string[] = [];
  if (s.withBoxes > 0) parts.push(`${s.withBoxes} 张图、${s.total} 个框`);
  if (s.empty > 0) parts.push(`${s.empty} 张没找到`);
  if (s.failed > 0) parts.push(`${s.failed} 张失败`);
  return parts.join(' · ');
}
