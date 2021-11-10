//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THE SOFTWARE IS PROVIDED �AS IS�, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
// THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//*********************************************************

#include "f1_datalogger/image_logging/winrt/SimpleCapture.h"
using namespace winrt;
using namespace Windows;
using namespace Windows::Foundation;
using namespace Windows::System;
using namespace Windows::Graphics;
using namespace Windows::Graphics::Capture;
using namespace Windows::Graphics::DirectX;
using namespace Windows::Graphics::DirectX::Direct3D11;
using namespace Windows::Foundation::Numerics;
using namespace Windows::UI;
using namespace Windows::UI::Composition;

deepf1::winrtcapture::SimpleCapture::SimpleCapture(
    IDirect3DDevice const& device,
    GraphicsCaptureItem const& item)
{
    m_item = item;
    m_device = device;

	// Set up 
    m_d3dDevice = GetDXGIInterfaceFromObject<ID3D11Device>(m_device);
    m_d3dDevice->GetImmediateContext(m_d3dContext.put());

	auto size = m_item.Size();

    m_swapChain = CreateDXGISwapChain(
        m_d3dDevice, 
		static_cast<uint32_t>(size.Width),
		static_cast<uint32_t>(size.Height),
        static_cast<DXGI_FORMAT>(DirectXPixelFormat::B8G8R8A8UIntNormalized),
        2);

	// Create framepool, define pixel format (DXGI_FORMAT_B8G8R8A8_UNORM), and frame size. 
    m_framePool = Direct3D11CaptureFramePool::Create(
        m_device,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
		size);
    m_session = m_framePool.CreateCaptureSession(m_item);
    m_session.IsCursorCaptureEnabled(false);
    m_lastSize = size;
	m_frameArrived = m_framePool.FrameArrived(auto_revoke, { this, &SimpleCapture::OnFrameArrived });
}

// Start sending capture frames
void deepf1::winrtcapture::SimpleCapture::StartCapture()
{
    CheckClosed();
    m_oclCtx=cv::directx::ocl::initializeContextFromD3D11Device(m_d3dDevice.get());
    m_session.StartCapture();
}

ICompositionSurface deepf1::winrtcapture::SimpleCapture::CreateSurface(
    Compositor const& compositor)
{
    CheckClosed();
    return CreateCompositionSurfaceForSwapChain(compositor, m_swapChain.get());
}

// Process captured frames
void deepf1::winrtcapture::SimpleCapture::Close()
{
    auto expected = false;
    if (m_closed.compare_exchange_strong(expected, true))
    {
		m_frameArrived.revoke();
		m_framePool.Close();
        m_session.Close();

        m_swapChain = nullptr;
        m_framePool = nullptr;
        m_session = nullptr;
        m_item = nullptr;
    }
}

void deepf1::winrtcapture::SimpleCapture::OnFrameArrived(
    Direct3D11CaptureFramePool const& sender,
    winrt::Windows::Foundation::IInspectable const&)
{
    auto newSize = false;

    {
        auto frame = sender.TryGetNextFrame();
		auto frameContentSize = frame.ContentSize();

        if (frameContentSize.Width != m_lastSize.Width ||
			frameContentSize.Height != m_lastSize.Height)
        {
            // The thing we have been capturing has changed size.
            // We need to resize our swap chain first, then blit the pixels.
            // After we do that, retire the frame and then recreate our frame pool.
            newSize = true;
            m_lastSize = frameContentSize;
            m_swapChain->ResizeBuffers(
                2, 
				static_cast<uint32_t>(m_lastSize.Width),
				static_cast<uint32_t>(m_lastSize.Height),
                static_cast<DXGI_FORMAT>(DirectXPixelFormat::B8G8R8A8UIntNormalized), 
                0);
        }
        std::chrono::duration<uint64_t, std::nano> dur = std::chrono::duration_cast< std::chrono::duration<uint64_t, std::nano> >(frame.SystemRelativeTime());
        std::chrono::high_resolution_clock::time_point stamp(dur);
        D3D11_TEXTURE2D_DESC desc;
        {
            winrt::com_ptr<ID3D11Texture2D> frameSurface = GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());
            winrt::com_ptr<ID3D11Texture2D> backBuffer;
            check_hresult(m_swapChain->GetBuffer(0, guid_of<ID3D11Texture2D>(), backBuffer.put_void()));
            m_d3dContext->CopyResource(backBuffer.get(), frameSurface.get());
            backBuffer->GetDesc(&desc);
            desc.Usage = D3D11_USAGE_STAGING;
            desc.BindFlags = 0;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.MiscFlags = 0;
            m_d3dDevice->CreateTexture2D(&desc, nullptr, m_stagingTexture.put());
            m_d3dContext->CopyResource(m_stagingTexture.get(), backBuffer.get());
            std::lock_guard<std::mutex> lk(m_imagemutex);
            m_d3dContext->Map(
                    m_stagingTexture.get(),
                    0,
                    D3D11_MAP_READ,
                    0,
                    &m_mapInfo);
            m_mat = cv::Mat(desc.Height,desc.Width, CV_8UC4, m_mapInfo.pData, m_mapInfo.RowPitch);   
        }
        cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
        cv::imshow("image", getCurrentImage());
        cv::waitKey(1);
    }

    DXGI_PRESENT_PARAMETERS presentParameters = { 0 };
    m_swapChain->Present1(1, 0, &presentParameters);

    if (newSize)
    {
        m_framePool.Recreate(
            m_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            m_lastSize);
    }
}

cv::Mat deepf1::winrtcapture::SimpleCapture::getCurrentImage()
{
    std::lock_guard<std::mutex> lk(m_imagemutex);
    cv::Mat rtn;
    m_mat.copyTo(rtn);
    return rtn;
}
