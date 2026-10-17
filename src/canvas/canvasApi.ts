/** 供快捷键等外部逻辑操作画布视图（避免把视图状态放进 React） */
export interface CanvasApi {
  fit: () => void;
  zoomBy: (factor: number) => void;
  setSpace: (down: boolean) => void;
  /** 画布上是否正开着框的标签菜单 —— 开着时全局快捷键要让位 */
  menuOpen: () => boolean;
}

export const canvasApi: { current: CanvasApi | null } = { current: null };
