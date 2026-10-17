import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "@radix-ui/react-slot"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  // Phase 2：字号 / 圆角 / 时长 / 缓动全部改走设计 token。
  // 旧版 text-sm(14px) + 硬编码 rounded-[12px] + duration-150 + 默认 ease，
  // 导致同一个按钮在不同调用点大小圆角各不相同，是「界面不统一」的源头之一。
  "inline-flex shrink-0 cursor-pointer items-center justify-center gap-2 rounded-[12px] text-body font-medium whitespace-nowrap outline-none transition-[background-color,background-image,color,box-shadow,border-color,transform,opacity] duration-fast ease-smooth active:scale-[0.97] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-45 aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        // Phase 9「厚重感」：主按钮加**顶部 1px 内高光**。
        // 上边缘被照亮 = 这是一块有体积的实体，不是一个贴上去的色块。
        // （内高光 + 垂直渐变 = 按钮质感的两层，缺一层就会退回「扁平色块」）
        default:
          "surface-brand text-primary-foreground [box-shadow:inset_0_1px_0_rgba(255,255,255,0.25),var(--shadow-brand)] hover:brightness-[1.06] hover:[box-shadow:inset_0_1px_0_rgba(255,255,255,0.25),var(--shadow-brand-lg)]",
        destructive:
          "bg-destructive text-white [box-shadow:inset_0_1px_0_rgba(255,255,255,0.2),0_6px_18px_rgba(224,72,61,0.28)] hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:bg-destructive/60 dark:focus-visible:ring-destructive/40",
        // 白底按钮没有「顶部高光」可用（白对白看不见），改让**底边内阴影**压一下：
        // 底边暗一点 = 有厚度。这是白色按钮唯一能造体积的写法。
        outline:
          "border border-border bg-surface-panel text-foreground [box-shadow:inset_0_-1px_0_rgba(13,18,32,0.04),var(--shadow-xs)] hover:border-input hover:bg-surface-hover hover:text-foreground",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/70",
        // ghost 是全站用得最多的变体（图标按钮几乎全是它）：
        // hover 走 surface-hover 令牌，禁用态用 opacity 而不是换颜色，
        // 这样禁用图标不会因为「变成另一种灰」而看起来像可点的。
        ghost:
          "text-ink-2 hover:bg-surface-hover hover:text-foreground dark:hover:bg-accent/50",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        xs: "h-6 gap-1 rounded-[9px] px-2 text-micro has-[>svg]:px-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 rounded-xl px-6 has-[>svg]:px-4",
        icon: "size-9",
        // 图标按钮统一圆角到 7px：图标方形的容器用 10px 会显得"胖"
        "icon-xs": "size-6 rounded-[9px] [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-8 rounded-[12px]",
        "icon-lg": "size-10 rounded-[14px]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
