#pragma once
#include <f1_datalogger/car_data/timestamped_image_data.h>
#include <opencv2/core/ocl.hpp>
#include <mutex>
#include <thread>
#include <f1_datalogger/image_logging/visibility_control.h>

namespace f1_datalogger
{
namespace image_logging
{
namespace winrt_capture
{

class F1_DATALOGGER_IMAGE_LOGGING_PUBLIC SimpleCapture
{
    friend class CaptureWrapper;
public:
    SimpleCapture(
        winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device,
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat pixelFormat);
    ~SimpleCapture() { Close(); }
private:

    void StartCapture();
    winrt::Windows::UI::Composition::ICompositionSurface CreateSurface(
        winrt::Windows::UI::Composition::Compositor const& compositor);

    bool IsCursorEnabled() { CheckClosed(); return m_session.IsCursorCaptureEnabled(); }
	void IsCursorEnabled(bool value) 
    { 
        CheckClosed(); 
        auto lock = m_lock.lock_exclusive(); 
        m_session.IsCursorCaptureEnabled(value);
    }
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem CaptureItem() { return m_item; }

    void SetPixelFormat(winrt::Windows::Graphics::DirectX::DirectXPixelFormat pixelFormat)
    {
        CheckClosed();
        auto lock = m_lock.lock_exclusive();
        m_pixelFormatUpdate = pixelFormat;
    }

    void Close();
    void destroyWindows();
    deepf1::TimestampedImageData getData() const
    {
        std::scoped_lock<std::mutex> lk(image_mutex);
        return deepf1::TimestampedImageData(current_mat.timestamp, current_mat.image.clone());
    }

    void OnFrameArrived(
        winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender,
        winrt::Windows::Foundation::IInspectable const& args);

    inline void CheckClosed()
    {
        if (m_closed.load() == true)
        {
            throw winrt::hresult_error(RO_E_CLOSED);
        }
    }

    void ResizeSwapChain();
    bool TryResizeSwapChain(winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame const& frame);
    bool TryUpdatePixelFormat();

private:
    LARGE_INTEGER m_performance_counter_frequency;
    deepf1::TimestampedImageData current_mat;
    cv::ocl::Context m_oclCtx;
    mutable std::mutex image_mutex;

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_item{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_framePool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_session{ nullptr };
    winrt::Windows::Graphics::SizeInt32 m_lastSize;

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
    winrt::com_ptr<ID3D11Device> m_device_native;
    winrt::com_ptr<IDXGISwapChain1> m_swapChain{ nullptr };
    winrt::com_ptr<ID3D11DeviceContext> m_d3dContext{ nullptr };
    winrt::Windows::Graphics::DirectX::DirectXPixelFormat m_pixelFormat;

    wil::srwlock m_lock;
    std::optional<winrt::Windows::Graphics::DirectX::DirectXPixelFormat> m_pixelFormatUpdate = std::nullopt;

    std::atomic<bool> m_closed = false;
    std::atomic<bool> m_captureNextImage = false;

    std::chrono::milliseconds lasttimemilli;

    deepf1::TimePoint t0;
};


}
}
}

