#pragma once
#include <winrt/Windows.UI.Composition.h>
#include <windows.ui.composition.interop.h>
#include <d2d1_1.h>
#include <f1_datalogger/image_logging/visibility_control.h>

namespace util::uwp
{
    inline auto F1_DATALOGGER_IMAGE_LOGGING_PUBLIC CreateCompositionGraphicsDevice(winrt::Windows::UI::Composition::Compositor const& compositor, ::IUnknown* device)
    {
        winrt::Windows::UI::Composition::CompositionGraphicsDevice graphicsDevice{ nullptr };
        auto compositorInterop = compositor.as<ABI::Windows::UI::Composition::ICompositorInterop>();
        winrt::com_ptr<ABI::Windows::UI::Composition::ICompositionGraphicsDevice> graphicsInterop;
        winrt::check_hresult(compositorInterop->CreateGraphicsDevice(device, graphicsInterop.put()));
        winrt::check_hresult(graphicsInterop->QueryInterface(winrt::guid_of<winrt::Windows::UI::Composition::CompositionGraphicsDevice>(),
            winrt::put_abi(graphicsDevice)));
        return graphicsDevice;
    }

    inline void F1_DATALOGGER_IMAGE_LOGGING_PUBLIC ResizeSurface(winrt::Windows::UI::Composition::CompositionDrawingSurface const& surface,
        winrt::Windows::Foundation::Size const& size)
    {
        auto surfaceInterop = surface.as<ABI::Windows::UI::Composition::ICompositionDrawingSurfaceInterop>();
        winrt::check_hresult(surfaceInterop->Resize({ static_cast<LONG>(std::round(size.Width)), static_cast<LONG>(std::round(size.Height)) }));
    }

    // Do the type conversion from long -> float without warnings
    inline D2D1::Matrix3x2F F1_DATALOGGER_IMAGE_LOGGING_PUBLIC Translation(const POINT& pt)
    {
        return D2D1::Matrix3x2F::Translation(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }

    inline auto F1_DATALOGGER_IMAGE_LOGGING_PUBLIC SurfaceBeginDraw(winrt::Windows::UI::Composition::CompositionDrawingSurface const& surface)
    {
        auto surfaceInterop = surface.as<ABI::Windows::UI::Composition::ICompositionDrawingSurfaceInterop>();
        winrt::com_ptr<ID2D1DeviceContext> context;
        POINT offset = {};
        winrt::check_hresult(surfaceInterop->BeginDraw(nullptr, __uuidof(context), context.put_void(), &offset));
        context->SetTransform(Translation(offset));
        return context;
    }

    inline void F1_DATALOGGER_IMAGE_LOGGING_PUBLIC SurfaceEndDraw(winrt::Windows::UI::Composition::CompositionDrawingSurface const& surface)
    {
        auto surfaceInterop = surface.as<ABI::Windows::UI::Composition::ICompositionDrawingSurfaceInterop>();
        winrt::check_hresult(surfaceInterop->EndDraw());
    }

    inline auto F1_DATALOGGER_IMAGE_LOGGING_PUBLIC CreateCompositionSurfaceForSwapChain(winrt::Windows::UI::Composition::Compositor const& compositor, ::IUnknown* swapChain)
    {
        winrt::Windows::UI::Composition::ICompositionSurface surface{ nullptr };
        auto compositorInterop = compositor.as<ABI::Windows::UI::Composition::ICompositorInterop>();
        winrt::com_ptr<ABI::Windows::UI::Composition::ICompositionSurface> surfaceInterop;
        winrt::check_hresult(compositorInterop->CreateCompositionSurfaceForSwapChain(swapChain, surfaceInterop.put()));
        winrt::check_hresult(surfaceInterop->QueryInterface(winrt::guid_of<winrt::Windows::UI::Composition::ICompositionSurface>(), winrt::put_abi(surface)));
        return surface;
    }
}
