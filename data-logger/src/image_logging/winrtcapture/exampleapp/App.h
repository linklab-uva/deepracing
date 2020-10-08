#pragma once
#include "f1_datalogger/image_logging/winrtcapture/SimpleCapture.h"
#include "SimpleImageEncoder.h"

class App
{
public:
    App(//winrt::Windows::UI::Composition::ContainerVisual root
        // ,winrt::Windows::Graphics::Capture::GraphicsCapturePicker capturePicker
        // ,winrt::Windows::Storage::Pickers::FileSavePicker savePicker
        );
    ~App();
    void PixelFormat(winrt::Windows::Graphics::DirectX::DirectXPixelFormat pixelFormat);

    bool IsCursorEnabled();
    void IsCursorEnabled(bool value);

    void StopCapture();

public:
    std::shared_ptr<SimpleCapture> m_capture{ nullptr };

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem StartCaptureFromWindowHandle(HWND hwnd);
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem StartCaptureFromMonitorHandle(HMONITOR hmon);
 //   winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Graphics::Capture::GraphicsCaptureItem> StartCaptureWithPickerAsync();
   // winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Storage::StorageFile> TakeSnapshotAsync();
    winrt::Windows::Graphics::DirectX::DirectXPixelFormat PixelFormat() { return m_pixelFormat; }

private:
    void StartCaptureFromItem(winrt::Windows::Graphics::Capture::GraphicsCaptureItem item);

private:
    winrt::Windows::System::DispatcherQueue m_mainThread{ nullptr };
    // winrt::Windows::UI::Composition::Compositor m_compositor{ nullptr };
    // winrt::Windows::UI::Composition::ContainerVisual m_root{ nullptr };
    // winrt::Windows::UI::Composition::SpriteVisual m_content{ nullptr };
    // winrt::Windows::UI::Composition::CompositionSurfaceBrush m_brush{ nullptr };

    winrt::Windows::Graphics::Capture::GraphicsCapturePicker m_capturePicker{ nullptr };
    winrt::Windows::Storage::Pickers::FileSavePicker m_savePicker{ nullptr };

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
    winrt::Windows::Graphics::DirectX::DirectXPixelFormat m_pixelFormat = winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized;

};