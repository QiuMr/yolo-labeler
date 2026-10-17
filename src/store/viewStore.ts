export interface ViewTransform {
  /** 图像像素 -> 屏幕 CSS 像素 */
  scale: number;
  /** 图像左上角在画布中的位置 */
  ox: number;
  oy: number;
}

const listeners = new Set<() => void>();
let view: ViewTransform = { scale: 1, ox: 0, oy: 0 };

/**
 * 视图状态单独放在 React 之外：
 * 平移/缩放每帧都会变，走外部 store 可以避免整棵组件树重渲染。
 */
export const viewStore = {
  get: (): ViewTransform => view,
  set: (next: ViewTransform): void => {
    if (next.scale === view.scale && next.ox === view.ox && next.oy === view.oy) return;
    view = next;
    for (const l of listeners) l();
  },
  subscribe: (listener: () => void): (() => void) => {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
};
