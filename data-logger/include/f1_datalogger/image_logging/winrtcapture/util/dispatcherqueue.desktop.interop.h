#pragma once
#include <winrt/Windows.System.h>
#include <dispatcherqueue.h>
#include <f1_datalogger/image_logging/visibility_control.h>

namespace util::desktop
{
    inline auto F1_DATALOGGER_IMAGE_LOGGING_PUBLIC CreateDispatcherQueueControllerForCurrentThread()
    {
        namespace abi = ABI::Windows::System;

        DispatcherQueueOptions options
        {
            sizeof(DispatcherQueueOptions),
            DQTYPE_THREAD_CURRENT,
            DQTAT_COM_NONE
        };

        winrt::Windows::System::DispatcherQueueController controller{ nullptr };
        winrt::check_hresult(CreateDispatcherQueueController(options, reinterpret_cast<abi::IDispatcherQueueController**>(winrt::put_abi(controller))));
        return controller;
    }
}