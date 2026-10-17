import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "h-8 w-full min-w-0 rounded-[12px] border border-input bg-card px-2.5 py-1 text-body text-foreground [box-shadow:var(--shadow-xs)] outline-none transition-[color,box-shadow,border-color] selection:bg-primary selection:text-primary-foreground placeholder:text-muted-foreground focus-visible:border-primary focus-visible:[box-shadow:0_0_0_3px_rgba(37,99,235,0.18)] disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-destructive aria-invalid:[box-shadow:0_0_0_3px_rgba(224,72,61,0.16)] file:inline-flex file:h-6 file:border-0 file:bg-transparent file:text-caption file:font-medium file:text-foreground",
        className
      )}
      {...props}
    />
  )
}

export { Input }
