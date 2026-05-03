; ImageViewer Installer Script
; Requires NSIS (Nullsoft Scriptable Install System) with MUI2

!define APP_NAME  "ImageViewer"
!define COMPANY   "Yakirma"
!define VERSION   "1.1.4"
!define REGKEY    "Software\${APP_NAME}"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

Name "${APP_NAME} ${VERSION}"
OutFile "dist\${APP_NAME}_Setup.exe"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "${REGKEY}" "Install_Dir"
RequestExecutionLevel admin

; ── Installer metadata (shown in file properties) ────────────────────────────
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName"     "${APP_NAME}"
VIAddVersionKey "CompanyName"     "${COMPANY}"
VIAddVersionKey "FileVersion"     "${VERSION}"
VIAddVersionKey "ProductVersion"  "${VERSION}"
VIAddVersionKey "FileDescription" "ImageViewer Installer"
VIAddVersionKey "LegalCopyright"  "© 2025 ${COMPANY}"

; ── UI ───────────────────────────────────────────────────────────────────────
!include "MUI2.nsh"
!define MUI_ABORTWARNING
!define MUI_ICON   "assets\icons\icon.ico"
!define MUI_UNICON "assets\icons\icon.ico"

; Install pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!define MUI_PAGE_CUSTOMFUNCTION_PRE  ComponentsPagePre
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstall pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Function ComponentsPagePre
  ; Nothing — placeholder so the define above compiles.
FunctionEnd

; ── Sections ─────────────────────────────────────────────────────────────────

Section "ImageViewer (required)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"

  ; Registry: install path + Add/Remove Programs entry
  WriteRegStr   HKLM "${REGKEY}" "Install_Dir" "$INSTDIR"
  WriteRegStr   HKLM "${UNINST_KEY}" "DisplayName"    "${APP_NAME}"
  WriteRegStr   HKLM "${UNINST_KEY}" "DisplayVersion"  "${VERSION}"
  WriteRegStr   HKLM "${UNINST_KEY}" "Publisher"       "${COMPANY}"
  WriteRegStr   HKLM "${UNINST_KEY}" "DisplayIcon"     "$INSTDIR\ImageViewer.exe,0"
  WriteRegStr   HKLM "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair"  1

  ; Copy application files
  File /r "dist\ImageViewer\*"

  ; ── Visual C++ Redistributable ───────────────────────────────────────────
  DetailPrint "Checking for Visual C++ 2015-2022 Redistributable..."
  ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  IntCmp $0 1 redist_done

  IfFileExists "$EXEDIR\vc_redist.x64.exe" 0 redist_missing
    DetailPrint "Installing Visual C++ 2015-2022 Redistributable..."
    ExecWait '"$EXEDIR\vc_redist.x64.exe" /install /quiet /norestart' $1
    DetailPrint "Redistributable installer exited with code $1"
    Goto redist_done

  redist_missing:
    DetailPrint "vc_redist.x64.exe not found next to installer — skipping."

  redist_done:

  ; Uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Start Menu shortcut
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut  "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" \
                  "$INSTDIR\ImageViewer.exe" "" "$INSTDIR\ImageViewer.exe" 0
  CreateShortCut  "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" \
                  "$INSTDIR\uninstall.exe"   "" "$INSTDIR\uninstall.exe"   0

  ; ── File Associations ───────────────────────────────────────────────────
  WriteRegStr HKCR "Applications\ImageViewer.exe" "" ""
  WriteRegStr HKCR "Applications\ImageViewer.exe" "FriendlyAppName"  "ImageViewer"
  WriteRegStr HKCR "Applications\ImageViewer.exe" "MultiSelectModel" "Player"
  WriteRegStr HKCR "Applications\ImageViewer.exe\shell\open\command" "" \
                   '"$INSTDIR\ImageViewer.exe" "%1"'

  ; Supported extensions in Open-With list
  !macro RegSupportedType EXT
    WriteRegStr HKCR "Applications\ImageViewer.exe\SupportedTypes" ".${EXT}" ""
  !macroend
  !insertmacro RegSupportedType "png"
  !insertmacro RegSupportedType "jpg"
  !insertmacro RegSupportedType "jpeg"
  !insertmacro RegSupportedType "tif"
  !insertmacro RegSupportedType "tiff"
  !insertmacro RegSupportedType "raw"
  !insertmacro RegSupportedType "bin"
  !insertmacro RegSupportedType "u16"
  !insertmacro RegSupportedType "f32"
  !insertmacro RegSupportedType "flo"
  !insertmacro RegSupportedType "npz"
  !insertmacro RegSupportedType "npy"
  !insertmacro RegSupportedType "gif"
  !insertmacro RegSupportedType "webp"
  !insertmacro RegSupportedType "heic"
  !insertmacro RegSupportedType "heif"
  !insertmacro RegSupportedType "mp4"
  !insertmacro RegSupportedType "avi"
  !insertmacro RegSupportedType "mov"
  !insertmacro RegSupportedType "mkv"
  !insertmacro RegSupportedType "webm"

  !macro RegisterExtension EXT DESC
    WriteRegStr HKCR "ImageViewer.${EXT}" ""                            "${DESC}"
    WriteRegStr HKCR "ImageViewer.${EXT}\DefaultIcon"                   "" "$INSTDIR\ImageViewer.exe,0"
    WriteRegStr HKCR "ImageViewer.${EXT}\shell"                         "" "open"
    WriteRegStr HKCR "ImageViewer.${EXT}\shell\open\command"            "" '"$INSTDIR\ImageViewer.exe" "%1"'
    WriteRegStr HKCR ".${EXT}\OpenWithProgIDs" "ImageViewer.${EXT}"     ""
  !macroend

  !insertmacro RegisterExtension "png"  "PNG Image"
  !insertmacro RegisterExtension "jpg"  "JPEG Image"
  !insertmacro RegisterExtension "jpeg" "JPEG Image"
  !insertmacro RegisterExtension "tif"  "TIFF Image"
  !insertmacro RegisterExtension "tiff" "TIFF Image"
  !insertmacro RegisterExtension "raw"  "Raw Image Data"
  !insertmacro RegisterExtension "bin"  "Binary Image Data"
  !insertmacro RegisterExtension "u16"  "16-bit Unsigned Image"
  !insertmacro RegisterExtension "f32"  "32-bit Float Image"
  !insertmacro RegisterExtension "flo"  "Optical Flow File"
  !insertmacro RegisterExtension "npz"  "NumPy Archive"
  !insertmacro RegisterExtension "npy"  "NumPy Array"
  !insertmacro RegisterExtension "gif"  "GIF Image"
  !insertmacro RegisterExtension "webp" "WebP Image"
  !insertmacro RegisterExtension "heic" "HEIC Image"
  !insertmacro RegisterExtension "heif" "HEIF Image"
  !insertmacro RegisterExtension "mp4"  "MPEG-4 Video"
  !insertmacro RegisterExtension "avi"  "AVI Video"
  !insertmacro RegisterExtension "mov"  "QuickTime Video"
  !insertmacro RegisterExtension "mkv"  "Matroska Video"
  !insertmacro RegisterExtension "webm" "WebM Video"

  ; Notify shell to refresh icons
  System::Call 'shell32::SHChangeNotify(i 0x8000000, i 0, i 0, i 0)'

SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortCut "$DESKTOP\${APP_NAME}.lnk" \
                 "$INSTDIR\ImageViewer.exe" "" "$INSTDIR\ImageViewer.exe" 0
SectionEnd

; ── Section descriptions (shown in components page) ──────────────────────────
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain}    "Core application files (required)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Create a shortcut on the Desktop."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ── Uninstaller ──────────────────────────────────────────────────────────────
Section "Uninstall"
  ; Registry cleanup
  DeleteRegKey HKLM "${UNINST_KEY}"
  DeleteRegKey HKLM "${REGKEY}"
  DeleteRegKey HKCR "Applications\ImageViewer.exe"

  !macro UnregisterExtension EXT
    DeleteRegKey  HKCR "ImageViewer.${EXT}"
    DeleteRegValue HKCR ".${EXT}\OpenWithProgIDs" "ImageViewer.${EXT}"
  !macroend

  !insertmacro UnregisterExtension "png"
  !insertmacro UnregisterExtension "jpg"
  !insertmacro UnregisterExtension "jpeg"
  !insertmacro UnregisterExtension "tif"
  !insertmacro UnregisterExtension "tiff"
  !insertmacro UnregisterExtension "raw"
  !insertmacro UnregisterExtension "bin"
  !insertmacro UnregisterExtension "u16"
  !insertmacro UnregisterExtension "f32"
  !insertmacro UnregisterExtension "flo"
  !insertmacro UnregisterExtension "npz"
  !insertmacro UnregisterExtension "npy"
  !insertmacro UnregisterExtension "gif"
  !insertmacro UnregisterExtension "webp"
  !insertmacro UnregisterExtension "heic"
  !insertmacro UnregisterExtension "heif"
  !insertmacro UnregisterExtension "mp4"
  !insertmacro UnregisterExtension "avi"
  !insertmacro UnregisterExtension "mov"
  !insertmacro UnregisterExtension "mkv"
  !insertmacro UnregisterExtension "webm"

  System::Call 'shell32::SHChangeNotify(i 0x8000000, i 0, i 0, i 0)'

  ; Shortcuts
  Delete "$DESKTOP\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"

  ; Application files
  RMDir /r "$INSTDIR"

SectionEnd
