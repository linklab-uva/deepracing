#pragma once
#include "f1_datalogger/image_logging/winrtcapture/util/DesktopWindow.h"

class WindowList;
class MonitorList;
namespace f1_datalogger
{
namespace image_logging
{
namespace winrt_capture
{
    class CaptureWrapper;
}
}
}
struct SimpleWindow : util::desktop::DesktopWindow<SimpleWindow>
{
    static const std::wstring ClassName;
    static void RegisterWindowClass();

    SimpleWindow(HINSTANCE instance, int cmdShow, std::shared_ptr<f1_datalogger::image_logging::winrt_capture::CaptureWrapper> app);
    ~SimpleWindow();

    winrt::Windows::UI::Composition::Desktop::DesktopWindowTarget CreateWindowTarget(winrt::Windows::UI::Composition::Compositor const& compositor)
    {
        return util::desktop::CreateDesktopWindowTarget(compositor, m_window, true);
    }

    void InitializeObjectWithWindowHandle(winrt::Windows::Foundation::IUnknown const& object)
    {
        auto initializer = object.as<util::desktop::IInitializeWithWindow>();
        winrt::check_hresult(initializer->Initialize(m_window));
    }

    LRESULT MessageHandler(UINT const message, WPARAM const wparam, LPARAM const lparam);

private:
    struct PixelFormatData
    {
        std::wstring Name;
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat PixelFormat;
    };

    enum class CaptureType
    {
        ProgrammaticWindow,
        ProgrammaticMonitor,
        Picker,
    };

    void CreateControls(HINSTANCE instance);
    void SetSubTitle(std::wstring const& text);
 //   winrt::fire_and_forget OnPickerButtonClicked();
  //  winrt::fire_and_forget OnSnapshotButtonClicked();
    void StopCapture();
    void OnCaptureItemClosed(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const&, winrt::Windows::Foundation::IInspectable const&);
    void OnCaptureStarted(
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item, 
        CaptureType captureType);

private:
    HWND m_windowComboBox = nullptr;
    HWND m_monitorComboBox = nullptr;
    HWND m_pickerButton = nullptr;
    HWND m_stopButton = nullptr;
    HWND m_snapshotButton = nullptr;
    HWND m_pixelFormatComboBox = nullptr;
    HWND m_cursorCheckBox = nullptr;
    HWND m_captureExcludeCheckBox = nullptr;
    std::unique_ptr<WindowList> m_windows;
    std::unique_ptr<MonitorList> m_monitors;
    std::vector<PixelFormatData> m_pixelFormats;
    std::shared_ptr<f1_datalogger::image_logging::winrt_capture::CaptureWrapper> m_app;
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem::Closed_revoker m_itemClosedRevoker;
};