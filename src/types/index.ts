/** 二维点 */
export interface IPoint {
  x: number;
  y: number;
}

/** 矩形（统一使用图像原始像素坐标存储） */
export interface IRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** 标签（类别） */
export interface Label {
  id: string;
  name: string;
  color: string;
}

/** 单个标注框 */
export interface Annotation {
  id: string;
  labelId: string;
  rect: IRect;
}

/** 一张图片 */
export interface ImageItem {
  id: string;
  name: string;
  file: File | null;
  width: number;
  height: number;
  annotations: Annotation[];
}

/** 导入结果：added 新增张数，skipped 因重复被跳过的张数，revived 重新接回原标注的张数 */
export interface AddImagesResult {
  added: number;
  skipped: number;
  /** 之前图片数据丢失、只剩标注，这次重新导入把图接了回去 */
  revived: number;
}

/** 导出格式 */
export type ExportFormat = 'yolo' | 'coco' | 'voc';

/** 导出配置 */
export interface ExportOptions {
  format: ExportFormat;
  /** 验证集比例 0 ~ 0.9，0 表示不划分 */
  splitRatio: number;
  /** 是否生成 data.yaml（仅 YOLO） */
  withYaml: boolean;
  /** 是否把图片一起打包进 zip */
  includeImages: boolean;
  /** 是否为没有标注的图片也生成空的标注文件 */
  exportEmpty: boolean;
}

/** 持久化：项目元信息 */
export interface PersistMeta {
  version: number;
  labels: Label[];
  activeLabelId: string;
  images: Array<{
    id: string;
    name: string;
    width: number;
    height: number;
    /**
     * 原始文件的修改时间。判重指纹是「名字 + 大小 + 修改时间」，
     * 而从 IndexedDB 重建 File 时不传 lastModified 会默认成 Date.now()，
     * 导致刷新一次判重就失效、同一批图被反复导入成重复条目。
     * 0 表示「未知」（旧版本数据没有这个字段）。
     */
    lastModified: number;
  }>;
}

/** 持久化：图片二进制 */
export interface PersistBlob {
  id: string;
  blob: Blob;
}

/** 持久化：图片标注 */
export interface PersistAnns {
  id: string;
  annotations: Annotation[];
}
