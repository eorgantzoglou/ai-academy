' AI Academy - shortcut and registry setup.
' Called by install.bat / uninstall.bat; not meant to be run directly.
'
'   cscript //nologo setup.vbs install "<install folder>"
'   cscript //nologo setup.vbs remove
'
' Written in VBScript rather than PowerShell so installing never depends on
' the machine's PowerShell execution policy.

Option Explicit

Const APP_NAME = "AI Academy"
Const APP_VER  = "1.0.0"
Const REG_KEY  = "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\AIAcademy\"

Dim sh, fso
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' ---------------------------------------------------------------- helpers --

Function RegGet(path)
  RegGet = ""
  On Error Resume Next
  RegGet = sh.RegRead(path)
  On Error GoTo 0
End Function

' Locates a Chromium-based browser so the app can open in a chromeless
' window (--app) instead of an ordinary tab. Returns "" if none is found.
Function FindBrowser()
  Dim names, roots, n, r, p, guesses
  names = Array("msedge.exe", "chrome.exe", "brave.exe")
  roots = Array("HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\", _
                "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\")
  For Each n In names
    For Each r In roots
      p = Replace(RegGet(r & n & "\"), """", "")
      If p <> "" Then
        If fso.FileExists(p) Then FindBrowser = p : Exit Function
      End If
    Next
  Next

  guesses = Array( _
    sh.ExpandEnvironmentStrings("%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"), _
    sh.ExpandEnvironmentStrings("%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"), _
    sh.ExpandEnvironmentStrings("%ProgramFiles%\Google\Chrome\Application\chrome.exe"), _
    sh.ExpandEnvironmentStrings("%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"), _
    sh.ExpandEnvironmentStrings("%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"))
  For Each p In guesses
    If fso.FileExists(p) Then FindBrowser = p : Exit Function
  Next
  FindBrowser = ""
End Function

Function FileUrl(p)
  FileUrl = "file:///" & Replace(Replace(p, "\", "/"), " ", "%20")
End Function

Function FolderSizeKB(path)
  On Error Resume Next
  FolderSizeKB = Int(fso.GetFolder(path).Size / 1024)
  If Err.Number <> 0 Then FolderSizeKB = 4096
  On Error GoTo 0
End Function

' ------------------------------------------------------------- shortcuts --

Sub MakeAppLink(linkPath, dest)
  Dim lnk, browser
  Set lnk = sh.CreateShortcut(linkPath)
  browser = FindBrowser()
  If browser <> "" Then
    ' Point straight at the browser so the shortcut can be pinned and the
    ' window belongs to the app rather than to a script host.
    lnk.TargetPath = browser
    lnk.Arguments  = "--app=""" & FileUrl(fso.BuildPath(dest, "index.html")) & """"
  Else
    lnk.TargetPath = sh.ExpandEnvironmentStrings("%SystemRoot%\System32\wscript.exe")
    lnk.Arguments  = """" & fso.BuildPath(dest, "launch.vbs") & """"
  End If
  lnk.WorkingDirectory = dest
  lnk.IconLocation     = fso.BuildPath(dest, "assets\ai-academy.ico") & ", 0"
  lnk.Description      = "AI Academy - Data Science, ML, Deep Learning and AI, offline"
  lnk.Save
End Sub

Sub MakeUninstallLink(linkPath, dest)
  Dim lnk
  Set lnk = sh.CreateShortcut(linkPath)
  lnk.TargetPath       = fso.BuildPath(dest, "uninstall.bat")
  lnk.WorkingDirectory = dest
  lnk.IconLocation     = sh.ExpandEnvironmentStrings("%SystemRoot%\System32\shell32.dll") & ", 31"
  lnk.Description      = "Remove AI Academy from this PC"
  lnk.Save
End Sub

Sub Kill(path)
  On Error Resume Next
  If fso.FileExists(path)   Then fso.DeleteFile path, True
  If fso.FolderExists(path) Then fso.DeleteFolder path, True
  On Error GoTo 0
End Sub

' ------------------------------------------------------------------ main --

Dim mode, dest, desktopLink, startDir
mode = ""
If WScript.Arguments.Count > 0 Then mode = LCase(WScript.Arguments(0))

desktopLink = fso.BuildPath(sh.SpecialFolders("Desktop"), APP_NAME & ".lnk")
startDir    = fso.BuildPath(sh.SpecialFolders("Programs"), APP_NAME)

If mode = "install" Then
  If WScript.Arguments.Count < 2 Then
    WScript.Echo "setup.vbs install needs the install folder"
    WScript.Quit 2
  End If
  dest = WScript.Arguments(1)
  If Not fso.FolderExists(dest) Then
    WScript.Echo "install folder does not exist: " & dest
    WScript.Quit 2
  End If

  MakeAppLink desktopLink, dest
  If Not fso.FolderExists(startDir) Then fso.CreateFolder startDir
  MakeAppLink       fso.BuildPath(startDir, APP_NAME & ".lnk"), dest
  MakeUninstallLink fso.BuildPath(startDir, "Uninstall " & APP_NAME & ".lnk"), dest

  ' Appears in Settings > Apps > Installed apps
  sh.RegWrite REG_KEY & "DisplayName",     APP_NAME, "REG_SZ"
  sh.RegWrite REG_KEY & "DisplayVersion",  APP_VER, "REG_SZ"
  sh.RegWrite REG_KEY & "Publisher",       "AI Academy", "REG_SZ"
  sh.RegWrite REG_KEY & "DisplayIcon",     fso.BuildPath(dest, "assets\ai-academy.ico"), "REG_SZ"
  sh.RegWrite REG_KEY & "InstallLocation", dest, "REG_SZ"
  sh.RegWrite REG_KEY & "UninstallString", """" & fso.BuildPath(dest, "uninstall.bat") & """", "REG_SZ"
  sh.RegWrite REG_KEY & "EstimatedSize",   FolderSizeKB(dest), "REG_DWORD"
  sh.RegWrite REG_KEY & "NoModify",        1, "REG_DWORD"
  sh.RegWrite REG_KEY & "NoRepair",        1, "REG_DWORD"

  Dim b
  b = FindBrowser()
  If b <> "" Then
    WScript.Echo "  shortcuts created (app window via " & fso.GetFileName(b) & ")"
  Else
    WScript.Echo "  shortcuts created (no Chromium browser found - will open in your default browser)"
  End If

ElseIf mode = "remove" Then
  Kill desktopLink
  Kill startDir
  On Error Resume Next
  sh.RegDelete REG_KEY
  On Error GoTo 0
  WScript.Echo "  shortcuts and registry entry removed"

Else
  WScript.Echo "usage: setup.vbs install ""<folder>""  |  setup.vbs remove"
  WScript.Quit 2
End If
