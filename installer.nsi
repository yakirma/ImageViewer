; ImageViewer Installer Script
; Requires NSIS (Nullsoft Scriptable Install System)

!define APP_NAME "ImageViewer"
!define COMPANY "Yakirma"
!define VERSION "1.0.0"

Name "${APP_NAME}"
OutFile "dist\${APP_NAME}_Setup.exe"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "Software\${APP_NAME}" "Install_Dir"
RequestExecutionLevel admin

; UI settings
!include "MUI2.nsh"
!define MUI_ABORTWARNING
!define MUI_ICON "assets\icons\icon.ico" 
!define MUI_UNICON "assets\icons\icon.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "ImageViewer (required)"
  SectionIn RO
  
  SetOutPath "$INSTDIR"
  
  ; Write reg keys
  WriteRegStr HKLM "SOFTWARE\${APP_NAME}" "Install_Dir" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoRepair" 1

  ; Files
  File /r "dist\ImageViewer\*"

  ; Uninstaller
  WriteUninstaller "uninstall.exe"
  
  ; Shortcuts
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\ImageViewer.exe"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\uninstall.exe"

  ; File Associations
  !macro RegisterExtension EXT DESC
    WriteRegStr HKCR ".${EXT}" "" "${APP_NAME}.${EXT}"
    WriteRegStr HKCR "${APP_NAME}.${EXT}" "" "${DESC}"
    WriteRegStr HKCR "${APP_NAME}.${EXT}\DefaultIcon" "" "$INSTDIR\ImageViewer.exe,0"
    WriteRegStr HKCR "${APP_NAME}.${EXT}\shell" "" "open"
    WriteRegStr HKCR "${APP_NAME}.${EXT}\shell\open\command" "" '"$INSTDIR\ImageViewer.exe" "%1"'
    
    ; Register for "Open With"
    WriteRegStr HKCR ".${EXT}\OpenWithProgIDs" "${APP_NAME}.${EXT}" ""
  !macroend

  !insertmacro RegisterExtension "png" "PNG Image"
  !insertmacro RegisterExtension "jpg" "JPEG Image"
  !insertmacro RegisterExtension "jpeg" "JPEG Image"
  !insertmacro RegisterExtension "tif" "TIFF Image"
  !insertmacro RegisterExtension "tiff" "TIFF Image"
  !insertmacro RegisterExtension "raw" "Raw Image Data"
  !insertmacro RegisterExtension "bin" "Binary Image Data"
  !insertmacro RegisterExtension "u16" "16-bit Unsigned Image"
  !insertmacro RegisterExtension "f32" "32-bit Float Image"
  !insertmacro RegisterExtension "mp4" "MPEG-4 Video"
  !insertmacro RegisterExtension "avi" "AVI Video"
  !insertmacro RegisterExtension "mov" "QuickTime Video"

SectionEnd

Section "Uninstall"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
  DeleteRegKey HKLM "SOFTWARE\${APP_NAME}"

  ; Remove File Associations
  !macro UnregisterExtension EXT
    DeleteRegKey HKCR "${APP_NAME}.${EXT}"
    DeleteRegValue HKCR ".${EXT}\OpenWithProgIDs" "${APP_NAME}.${EXT}"
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
  !insertmacro UnregisterExtension "mp4"
  !insertmacro UnregisterExtension "avi"
  !insertmacro UnregisterExtension "mov"

  RMDir /r "$SMPROGRAMS\${APP_NAME}"
  RMDir /r "$INSTDIR"
SectionEnd
