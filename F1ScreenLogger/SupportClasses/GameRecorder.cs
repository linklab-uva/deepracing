using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Threading;
namespace SupportClasses
{
    [Serializable()]
    public class GameRecorder
    {
        private LinkedList<TimeStampedScreencap> recording_;
        private Size size_;
        private TimeSpan rate_;
        private bool runnin_;
        private Thread thread_;
        public GameRecorder(Size size, TimeSpan rate)
        {
            recording_ = new LinkedList<TimeStampedScreencap>();
            size_ = size;
            rate_ = rate;
            thread_ = new Thread(delegate () { this.recordWorker(); });
        }
        public void init(uint num_elements)
        {
            recording_.Clear();
            for(uint i = 0; i < num_elements; i++)
            {
                recording_.AddLast(new TimeStampedScreencap(size_));
            }
        }
        public void stop()
        {
            runnin_ = false;
        }
        public void record()
        {
            thread_.Start();
        }
        public void recordWorker()
        {
            LinkedListNode<TimeStampedScreencap> node = recording_.First;
            while (runnin_ && node.Next!=null)
            {
                node.Value.capture();
                node = node.Next;
                Thread.Sleep(rate_);
            }
        }
    }
}
