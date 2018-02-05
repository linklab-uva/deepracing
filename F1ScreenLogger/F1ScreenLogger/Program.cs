using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SupportClasses;
using System.Drawing;
namespace F1ScreenLogger
{
    class Program
    {
        static void Main(string[] args)
        {
            GameRecorder gr = new GameRecorder(new Size(700,700), TimeSpan.FromMilliseconds(20.0));
            gr.init((uint)int.Parse(args[0]));
            Console.Read();
        }
    }
}
