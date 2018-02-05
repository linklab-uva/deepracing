using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
namespace SupportClasses
{
    [Serializable()]
    public class TimeStampedScreencap
    {
        private Bitmap img_;
        private Size size_;
        private Graphics graphics_;
        private DateTime timestamp_;
        public TimeStampedScreencap(Size size)
        {
            size_ = size;
            img_ = new Bitmap(size.Width, size.Height);
            graphics_ = Graphics.FromImage(img_);
        }
        public void capture()
        {
            graphics_.CopyFromScreen(0, 0, 0, 0, size_);
            timestamp_ = DateTime.Now;
        }
        
    }
}
