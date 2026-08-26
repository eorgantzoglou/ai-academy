' AI Academy launcher.
' Opens index.html in a chromeless app window if a Chromium browser is
' available, otherwise in the default browser. Used as the shortcut target
' when no Chromium browser was found at install time, and as a general
' double-click entry point. No console window flashes.

Option Explicit

Dim sh, fso, here, page
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

here = fso.GetParentFolderName(WScript.ScriptFullName)
page = fso.BuildPath(here, "index.html")

If Not fso.FileExists(page) Then
  MsgBox "AI Academy could not find index.html next to this launcher:" & vbCrLf & vbCrLf & _
         here, 16, "AI Academy"
  WScript.Quit 1
End If

Function RegGet(path)
  RegGet = ""
  On Error Resume Next
  RegGet = sh.RegRead(path)
  On Error GoTo 0
End Function

' Kept in step with tools/setup.vbs, which does the same lookup at install time.
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

Dim exe, url
exe = FindBrowser()
url = "file:///" & Replace(Replace(page, "\", "/"), " ", "%20")

If exe <> "" Then
  sh.Run """" & exe & """ --app=""" & url & """", 1, False
Else
  sh.Run """" & page & """", 1, False
End If
