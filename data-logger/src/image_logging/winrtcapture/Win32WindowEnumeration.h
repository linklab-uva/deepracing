#pragma once
#include <dwmapi.h>
#include <codecvt>
namespace deepf1
{
namespace winrt_capture
{
struct Window
{
public:
    Window(nullptr_t) {}
    Window(HWND hwnd, std::wstring const& title, std::wstring& className)
    {
        m_hwnd = hwnd;
        m_title = title;
        m_className = className;
        RECT imrect;
        GetWindowRect(hwnd,&imrect);
        m_rows = imrect.right - imrect.left;
        m_cols = imrect.bottom - imrect.top;
        
    }

    HWND Hwnd() const noexcept { return m_hwnd; }
    std::wstring Title() const noexcept { return m_title; }
    std::string TitleStr() const noexcept {std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter; return std::string(converter.to_bytes(m_title)); }
    std::wstring ClassName() const noexcept { return m_className; }
    uint32_t rows() const noexcept { return m_rows; }
    uint32_t cols() const noexcept { return m_cols; }


private:
    HWND m_hwnd;
    std::wstring m_title;
    std::wstring m_className;
    uint32_t m_rows, m_cols;
};
}
}
std::wstring GetClassName(HWND hwnd)
{
	std::array<WCHAR, 1024> className;

    ::GetClassName(hwnd, className.data(), (int)className.size());

    std::wstring title(className.data());

    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return title;
}

std::wstring GetWindowText(HWND hwnd)
{
	std::array<WCHAR, 1024> windowText;

    ::GetWindowText(hwnd, windowText.data(), (int)windowText.size());

    std::wstring title(windowText.data());

    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return title;
}


bool IsAltTabWindow(deepf1::winrt_capture::Window const& window)
{
    HWND hwnd = window.Hwnd();
    HWND shellWindow = GetShellWindow();

    auto title = window.Title();
    auto className = window.ClassName();

    if (hwnd == shellWindow)
    {
        return false;
    }

    if (title.length() == 0)
    {
        return false;
    }

    if (!IsWindowVisible(hwnd))
    {
        return false;
    }

    if (GetAncestor(hwnd, GA_ROOT) != hwnd)
    {
        return false;
    }

    LONG style = GetWindowLong(hwnd, GWL_STYLE);
    if (!((style & WS_DISABLED) != WS_DISABLED))
    {
        return false;
    }

    DWORD cloaked = FALSE;
    HRESULT hrTemp = DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
    if (SUCCEEDED(hrTemp) &&
        cloaked == DWM_CLOAKED_SHELL)
    {
        return false;
    }

    return true;
}

BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam)
{
    auto class_name = GetClassName(hwnd);
    auto title = GetWindowText(hwnd);

    auto window = deepf1::winrt_capture::Window(hwnd, title, class_name);

    if (!IsAltTabWindow(window))
    {
        return TRUE;
    }

    std::vector<deepf1::winrt_capture::Window>& windows = *reinterpret_cast<std::vector<deepf1::winrt_capture::Window>*>(lParam);
    windows.push_back(window);

    return TRUE;
}

const std::vector<deepf1::winrt_capture::Window> EnumerateWindows()
{
    std::vector<deepf1::winrt_capture::Window> windows;
    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&windows));

    return windows;
}