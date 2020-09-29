#pragma once

#include <f1_datalogger/image_logging/visibility_control.h>

class F1_DATALOGGER_IMAGE_LOGGING_PUBLIC SimpleCapture;

class F1_DATALOGGER_IMAGE_LOGGING_PUBLIC CaptureApp
{
public:
    CaptureApp() {}
    ~CaptureApp() {}

    void Initialize(
        winrt::Windows::UI::Composition::ContainerVisual const& root);

    void StartCapture(HWND hwnd);

private:
    winrt::Windows::UI::Composition::Compositor m_compositor{ nullptr };
    winrt::Windows::UI::Composition::ContainerVisual m_root{ nullptr };
    winrt::Windows::UI::Composition::SpriteVisual m_content{ nullptr };
    winrt::Windows::UI::Composition::CompositionSurfaceBrush m_brush{ nullptr };

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
    std::unique_ptr<SimpleCapture> m_capture{ nullptr };
};