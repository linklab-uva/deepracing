#pragma once
#include <winrt/Windows.UI.Composition.h>
#include <winrt/Windows.UI.Composition.Desktop.h>
#include <windows.ui.composition.interop.h>
#include <f1_datalogger/image_logging/visibility_control.h>

namespace util::desktop
{
    inline auto F1_DATALOGGER_IMAGE_LOGGING_PUBLIC CreateDesktopWindowTarget(winrt::Windows::UI::Composition::Compositor const& compositor, HWND window, bool isTopMost)
    {
        namespace abi = ABI::Windows::UI::Composition::Desktop;

        auto interop = compositor.as<abi::ICompositorDesktopInterop>();
        winrt::Windows::UI::Composition::Desktop::DesktopWindowTarget target{ nullptr };
        winrt::check_hresult(interop->CreateDesktopWindowTarget(window, isTopMost, reinterpret_cast<abi::IDesktopWindowTarget**>(winrt::put_abi(target))));
        return target;
    }
}