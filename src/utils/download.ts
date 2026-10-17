/** 触发浏览器下载，objectURL 延迟释放，兼容各浏览器 */
export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 1000);
}

export function textBlob(content: string, type = 'text/plain;charset=utf-8'): Blob {
  return new Blob([content], { type });
}
