# Generates assets/ai-academy.ico (multi-resolution) and assets/icon-256.png
# from code, so the repository needs no binary art committed by hand.
#   Run:  powershell -ExecutionPolicy Bypass -File tools\make-icon.ps1
Add-Type -AssemblyName System.Drawing

$root   = Split-Path -Parent $PSScriptRoot
$assets = Join-Path $root 'assets'
if (-not (Test-Path $assets)) { New-Item -ItemType Directory -Path $assets | Out-Null }

# Brand palette, taken from css/style.css
$ACCENT   = [System.Drawing.ColorTranslator]::FromHtml('#6ea8ff')
$ACCENT_2 = [System.Drawing.ColorTranslator]::FromHtml('#8b7bff')

function New-RoundedPath([single]$x, [single]$y, [single]$w, [single]$h, [single]$r) {
  $p = New-Object System.Drawing.Drawing2D.GraphicsPath
  $d = $r * 2
  $p.AddArc($x,        $y,        $d, $d, 180, 90)
  $p.AddArc($x+$w-$d,  $y,        $d, $d, 270, 90)
  $p.AddArc($x+$w-$d,  $y+$h-$d,  $d, $d,   0, 90)
  $p.AddArc($x,        $y+$h-$d,  $d, $d,  90, 90)
  $p.CloseFigure()
  return $p
}

function New-IconBitmap([int]$S) {
  $bmp = New-Object System.Drawing.Bitmap($S, $S, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
  $g   = [System.Drawing.Graphics]::FromImage($bmp)
  $g.SmoothingMode     = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $g.PixelOffsetMode   = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
  $g.Clear([System.Drawing.Color]::Transparent)

  # everything below is authored in a 100x100 space, then scaled to S
  $u = $S / 100.0
  function P([single]$a, [single]$b) {
    New-Object System.Drawing.PointF(($a * $u), ($b * $u))
  }

  # --- the tile: brand gradient in a squircle ------------------------------
  $inset = 3 * $u
  $tile  = New-RoundedPath $inset $inset ($S - 2*$inset) ($S - 2*$inset) ($S * 0.235)
  $rect  = New-Object System.Drawing.RectangleF(0, 0, $S, $S)
  $grad  = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $ACCENT, $ACCENT_2, 45.0)
  $g.FillPath($grad, $tile)

  # a soft highlight along the top edge keeps it from looking flat
  $hi = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
          $rect,
          [System.Drawing.Color]::FromArgb(70, 255, 255, 255),
          [System.Drawing.Color]::FromArgb(0, 255, 255, 255), 90.0)
  $g.FillPath($hi, $tile)

  $white = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)

  # --- the mortarboard (reads clearly even at 16px) ------------------------
  $board = New-Object System.Drawing.Drawing2D.GraphicsPath
  $board.AddPolygon(@((P 50 22), (P 91 39), (P 50 56), (P 9 39)))
  $g.FillPath($white, $board)

  # --- the cap underneath --------------------------------------------------
  $cap = New-Object System.Drawing.Drawing2D.GraphicsPath
  $cap.AddLine((P 26 43), (P 26 62))
  $cap.AddBezier((P 26 62), (P 36 72), (P 64 72), (P 74 62))
  $cap.AddLine((P 74 62), (P 74 43))
  $cap.AddLine((P 74 43), (P 50 53))
  $cap.CloseFigure()
  $g.FillPath($white, $cap)

  # a thin gap between board and cap so the two shapes stay distinguishable
  $gapPen = New-Object System.Drawing.Pen($ACCENT_2, [single](2.6 * $u))
  $gapPen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
  $gap = New-Object System.Drawing.Drawing2D.GraphicsPath
  $gap.AddLine((P 26 43), (P 50 53))
  $gap.AddLine((P 50 53), (P 74 43))
  $g.DrawPath($gapPen, $gap)

  # --- tassel: only at sizes where a 2px cord survives ---------------------
  if ($S -ge 32) {
    $pen = New-Object System.Drawing.Pen($white.Color, [single](3.2 * $u))
    $pen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $pen.EndCap   = [System.Drawing.Drawing2D.LineCap]::Round
    $g.DrawLine($pen, (P 87 41), (P 87 68))
    $r = 5.0 * $u
    $g.FillEllipse($white, [single](87*$u - $r), [single](68*$u - $r), [single]($r*2), [single]($r*2))
    $pen.Dispose()
  }

  $g.Dispose(); $grad.Dispose(); $hi.Dispose(); $white.Dispose(); $gapPen.Dispose()
  $tile.Dispose(); $board.Dispose(); $cap.Dispose(); $gap.Dispose()
  return $bmp
}

function Get-PngBytes([System.Drawing.Bitmap]$bmp) {
  $ms = New-Object System.IO.MemoryStream
  $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
  $bytes = $ms.ToArray()
  $ms.Dispose()
  return ,$bytes
}

# A BITMAPINFOHEADER + bottom-up BGRA payload + AND mask, i.e. the classic
# icon format. Small frames use this rather than PNG because several
# consumers (GDI+ among them) decode PNG frames badly at 16-64px.
function Get-DibBytes([System.Drawing.Bitmap]$bmp) {
  $w = $bmp.Width; $h = $bmp.Height
  $rect = New-Object System.Drawing.Rectangle(0, 0, $w, $h)
  $data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                        [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
  $stride = [math]::Abs($data.Stride)
  $raw = New-Object byte[] ($stride * $h)
  [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $raw, 0, $raw.Length)
  $bmp.UnlockBits($data)

  $ms = New-Object System.IO.MemoryStream
  $bw = New-Object System.IO.BinaryWriter($ms)
  $bw.Write([uint32]40)          # biSize
  $bw.Write([int32]$w)           # biWidth
  $bw.Write([int32]($h * 2))     # biHeight: colour rows + mask rows
  $bw.Write([uint16]1)           # biPlanes
  $bw.Write([uint16]32)          # biBitCount
  $bw.Write([uint32]0)           # biCompression = BI_RGB
  $bw.Write([uint32]0)           # biSizeImage
  $bw.Write([int32]0); $bw.Write([int32]0)
  $bw.Write([uint32]0); $bw.Write([uint32]0)

  for ($y = $h - 1; $y -ge 0; $y--) { $bw.Write($raw, $y * $stride, $w * 4) }

  # AND mask: unused for 32bpp frames, but the row space must still be there
  $maskStride = [math]::Floor((($w + 31) / 32)) * 4
  $bw.Write((New-Object byte[] ($maskStride * $h)))

  $bw.Flush()
  $bytes = $ms.ToArray()
  $bw.Dispose(); $ms.Dispose()
  return ,$bytes
}

# --- pack the sizes into a single .ico -------------------------------------
# ICO layout: 6-byte header, then one 16-byte directory entry per image,
# then the image payloads. BMP payloads up to 64px for compatibility;
# PNG above that, which keeps 128 and 256 from bloating the file.
$sizes  = @(16, 20, 24, 32, 40, 48, 64, 128, 256)
$images = @()
foreach ($s in $sizes) {
  $bmp = New-IconBitmap $s
  if ($s -eq 256) { $bmp.Save((Join-Path $assets 'icon-256.png'), [System.Drawing.Imaging.ImageFormat]::Png) }
  if ($s -eq 64)  { $bmp.Save((Join-Path $assets 'icon-64.png'),  [System.Drawing.Imaging.ImageFormat]::Png) }
  if ($s -le 64) { $images += ,(Get-DibBytes $bmp) } else { $images += ,(Get-PngBytes $bmp) }
  $bmp.Dispose()
}

$ms = New-Object System.IO.MemoryStream
$bw = New-Object System.IO.BinaryWriter($ms)
$bw.Write([uint16]0)                  # reserved
$bw.Write([uint16]1)                  # type: 1 = icon
$bw.Write([uint16]$sizes.Count)

$offset = 6 + (16 * $sizes.Count)
for ($i = 0; $i -lt $sizes.Count; $i++) {
  $s = $sizes[$i]
  $bw.Write([byte]($(if ($s -ge 256) { 0 } else { $s })))   # 0 means 256
  $bw.Write([byte]($(if ($s -ge 256) { 0 } else { $s })))
  $bw.Write([byte]0)                  # palette size
  $bw.Write([byte]0)                  # reserved
  $bw.Write([uint16]1)                # colour planes
  $bw.Write([uint16]32)               # bits per pixel
  $bw.Write([uint32]$images[$i].Length)
  $bw.Write([uint32]$offset)
  $offset += $images[$i].Length
}
foreach ($img in $images) { $bw.Write($img) }
$bw.Flush()

$icoPath = Join-Path $assets 'ai-academy.ico'
[System.IO.File]::WriteAllBytes($icoPath, $ms.ToArray())
$bw.Dispose(); $ms.Dispose()

$kb = [math]::Round((Get-Item $icoPath).Length / 1KB, 1)
Write-Output "wrote assets/ai-academy.ico  ($($sizes.Count) sizes, $kb KB)"
Write-Output "wrote assets/icon-256.png, assets/icon-64.png"
