/** COCO-80 类别名（按模型输出 cls 索引）。绝大多数通用检测模型都是这套 */
export const COCO_NAMES: readonly string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

/**
 * 类别显示名：
 * 80 类模型按 COCO 表显示英文名；自定义模型只能显示类别序号。
 * 这个名字同时也是「采纳后」自动创建的标签名。
 */
export function classNameFor(classIndex: number, numClasses: number): string {
  if (numClasses === COCO_NAMES.length) {
    return COCO_NAMES[classIndex] ?? `类别 ${classIndex + 1}`;
  }
  return `类别 ${classIndex + 1}`;
}
