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
    Window(HWND hwnd, std::string const& title, std::string& className)
    {
        m_hwnd = hwnd;
        m_title = title;
        m_className = className;
        RECT rect;
        GetWindowRect(hwnd, &rect);
        m_cols = rect.right - rect.left;
        m_rows = rect.bottom - rect.top;
    }

    HWND Hwnd() const noexcept { return m_hwnd; }
    std::string Title() const noexcept { return m_title; }
    std::string ClassName() const noexcept { return m_className; }
    uint32_t rows() const noexcept { return m_rows; }
    uint32_t cols() const noexcept { return m_cols; }

private:
    HWND m_hwnd;
    std::string m_title;
    std::string m_className;
    uint32_t m_rows;
    uint32_t m_cols;
};
  
}
}

std::string GetClassName(HWND hwnd)
{
	std::array<WCHAR, 1024> className;

    ::GetClassNameW(hwnd, className.data(), (int)className.size());

    std::wstring title(className.data());

    //setup converter
    std::wstring_convert< std::codecvt_utf8<wchar_t> , wchar_t> converter;
    return std::string(converter.to_bytes(title));
}

std::string GetWindowText(HWND hwnd)
{
	std::array<WCHAR, 1024> windowText;

    ::GetWindowTextW(hwnd, windowText.data(), (int)windowText.size());

    std::wstring title(windowText.data());
    //setup converter
    std::wstring_convert< std::codecvt_utf8<wchar_t> , wchar_t> converter;
    return std::string(converter.to_bytes(title));
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