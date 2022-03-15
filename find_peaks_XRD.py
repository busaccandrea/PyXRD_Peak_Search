from turtle import color
from click import open_file
from matplotlib.cbook import index_of
import scipy.signal as sci_sig
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from matplotlib.widgets import Slider, Button, SpanSelector
from tkinter import Tk    # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from numpy import minimum,fft,pad
from scipy.signal import windows
from os import path

def snip(z,m):
    x = z.copy()
    for p in range(1,m)[::-1]:
        a1 = x[p:-p]
        a2 = (x[:(-2 * p)] + x[(2 * p):]) * 0.5
        x[p:-p] = minimum(a2,a1)

    return x

def convolve(z,n=21,std=3):
    kernel = windows.gaussian(2 * n - 1,std)

    y_pad = pad(z,(n,n),'edge')

    f = fft.rfft(y_pad)
    w = fft.rfft(kernel,y_pad.shape[-1])
    y = fft.irfft(w * f)

    return y[n * 2:] / sum(kernel)

class XRD_Peak_search_window:
    def __init__(self) -> None:

        self.filter_size = 20
        self.min_width = 2
        self.max_width = 50
        self.width_default = 10

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('angles')
        self.ax.set_ylabel('intensity')

        plt.subplots_adjust(left=0.25, bottom=0.35)
        
        self.init_spectrum()
        self.init_widgets()

        self.ax.plot(self.channels, self.spectrum, label='spectrum')
        self.ax.plot(self.channels, self.background, label='background')
        self.ax.plot(self.channels, self.net_spectrum, label='net spectrum')
        self.vlines = self.ax.vlines(self.channels[self.peaks[0]], 0, self.max_val, 'r', label='peak position')

    def init_spectrum(self):
        filename = self.open_new_file()

        with open(filename,'r') as f:
            lines = f.readlines()

        channels, spectrum = [],[]
        for line in lines:
            ch, intensity = line.split(' ')
            spectrum += [int(intensity)]
            channels += [float(ch)]

        self.spectrum_filename = filename
        self.spectrum = np.array(spectrum)
        self.channels = np.array(channels)
        self.width_list = np.linspace(self.width_default, 50, 10)
        self.spectrum = self.normalize(self.spectrum)
        self.max_val = np.max(self.spectrum)
        
        filter_size = 20
        self.mask=np.ones((1,filter_size))/filter_size
        self.mask=self.mask[0,:]
        self.background = snip(convolve(self.spectrum, n=35, std=3), m=16)
        self.net_spectrum = self.spectrum - self.background
        self.net_spectrum[self.net_spectrum<0]=0
        self.net_spectrum = np.convolve(self.net_spectrum, self.mask, 'same')

        self.net_spectrum = self.normalize(self.net_spectrum)

        # find peaks with default settings.
        self.peaks = sci_sig.find_peaks(np.log(self.net_spectrum), width=0.2, height=1)

        # once channels is initialized we can set lims on x axis
        self.ax.set_xlim(self.channels[0],self.channels[-1])

    def open_new_file(self):
        Tk().withdraw()
        filename = askopenfilename(filetypes=(('dat files', '*.dat'),('All files', '*.*')), initialdir='./')
        return filename

    def init_widgets(self):
        self.axwidth_min = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.width_slider_min = Slider(
            ax=self. axwidth_min,
            label='width min',
            valmin=0,
            valmax=50,
            valstep = 0.1,
            valinit=10
        )

        """ self.axwidth_max = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.width_slider_max = Slider(
            ax=self.axwidth_max,
            label='width max',
            valmin=0,
            valmax=50,
            valstep = 0.1,
            valinit=50
        )
 """
  
        self.axbgheight = plt.axes([0.25, 0.20, 0.215, 0.03])
        self.bgheight_slider = Slider(
        ax=self.axbgheight,
        label='background height',
        valmin=0,
        valmax=400,
        valstep = 1,
        valinit=0
        )
        
        self.axm = plt.axes([0.65, 0.20, 0.215, 0.03])
        self.m_slider = Slider(
        ax=self.axm,
        label='snip width',
        valmin=10,
        valmax=50,
        valstep = 1,
        valinit=16
    )

        self.openfileax = plt.axes([0.001, 0.90, 0.16, 0.04])
        self.openfilebutton = Button(self.openfileax, 'Open file', hovercolor='0.975')

        self.updateax = plt.axes([0.001, 0.855, 0.16, 0.04])
        self.updatebutton = Button(self.updateax, 'update plot', hovercolor='0.975')
        
        self.saveax = plt.axes([0.8, 0.025, 0.16, 0.04])
        self.savebutton = Button(self.saveax, 'Save peaks', hovercolor='0.975')
    
    def set_spectrum(self, filename):
        self.spectrum_filename = filename
        with open(filename,'r') as f: lines = f.readlines()
        self.channels, self.spectrum = [],[]

        for line in lines:
            ch, intensity = line.split(' ')
            self.spectrum += [int(intensity)]
            self.channels += [float(ch)]

        self.spectrum = self.normalize(np.array(self.spectrum))
        self.channels = np.array(self.channels)
        self.max_val = np.max(self.spectrum)

    def set_background(self):
        self.background = snip(convolve(self.spectrum, n=35, std=3), m=self.m_slider.val)
        self.background = self.background + self.bgheight_slider.val
        self.net_spectrum = self.spectrum - self.background
        
        self.net_spectrum[self.net_spectrum<0] = 0
        self.net_spectrum = np.convolve(self.net_spectrum, self.mask, 'same')
        self.max_val= np.max(self.net_spectrum)
        self.net_spectrum = self.normalize(self.net_spectrum)

    def set_peaks(self):
        sci_sig.find_peaks(np.log(self.net_spectrum), width=self.width_default, height=1)

    def update_peaks(self):
        p = []
        for width in self.width_list:
            peaks = sci_sig.find_peaks(np.log(self.net_spectrum), width=width, height=2)
            p += list(peaks[0])
        self.peaks = np.array(list(dict.fromkeys(p)))

    def normalize(self, arr):
        arr = 1000 * arr / np.max(arr)
        return arr

    def execute(self):
        def update_plot():
            self.ax.clear()
            self.ax.set_title(path.basename(self.spectrum_filename))

            self.ax.plot(self.channels, self.spectrum, label='raw', marker='.', lw=2)
            self.ax.plot(self.channels, self.background, label='background', lw=2)
            self.ax.plot(self.channels, self.net_spectrum, label='net spectrum', marker='.', lw=2)

            self.ax.vlines(self.channels[self.peaks], 0, 1000, 'r', label='peak position')
            self.ax.legend(frameon=True)
            self.ax.set_xlim(self.channels[0],self.channels[-1])
            self.fig.canvas.draw_idle()
        
        def update_background(val):
            self.set_background()
            self.update_peaks()

            update_plot()

        def width_slider_changed(val):
            # I have to update the widthlist
            # self.width_list = np.arange(self.width_slider_min.val, self.width_slider_max.val, 0.1)
            self.width_list = np.arange(self.width_slider_min.val, 50, 0.1)
            # I have to update peaks
            self.update_peaks()
            update_plot()

        def open_file(event):
            Tk().withdraw()
            filename = askopenfilename(filetypes=(('dat files', '*.dat'),('All files', '*.*')))
            self.set_spectrum(filename)
            self.set_background()
            self.update_peaks()
            update_plot()

        def save_peak(event):
            # savefig
            extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(self.spectrum_filename+'.png', bbox_inches=extent.expanded(1.1, 1.2))
            self.net_spectrum = self.normalize(self.net_spectrum)

            # save peaks: in each row of a file write position and intensity of a peak
            with open(self.spectrum_filename+'.txt', 'w') as outfile:
                for i, channel in enumerate(self.peaks):
                    outfile.write(str(self.channels[channel]) + ' ' + format(self.net_spectrum[channel], '.2f') +'\n')
        
        def update(event):
            update_plot()

        def onselect_remove(xmin, xmax):
            # take all peaks between min and max and remove them
            self.peaks = np.array(self.peaks)
            peak_energy_pos = self.channels[self.peaks]
            print('PEAKS',self.channels[self.peaks], xmax)
            peak2remove = peak_energy_pos[peak_energy_pos>=xmin]
            peak2remove = peak2remove[peak2remove<=xmax]

            print('PEAK2REMOVE',np.where(peak_energy_pos==peak2remove))
            for p in np.where(peak_energy_pos==peak2remove):
                self.peaks = np.delete(self.peaks, p)
            
            update_plot()

        def onselect_add(xmin, xmax):
            # take the position where the max is and put a peak

            # from self.net_spectrum take the range between xmin and xmax
            energy_range = self.channels[self.channels>=xmin]
            energy_range = energy_range[energy_range<=xmax]
            tmpmax = 0
            for e in energy_range:
                ind = np.where(self.channels==e)
                if self.net_spectrum[ind]>tmpmax: 
                    tmpmax=self.net_spectrum[ind]
                    indmax=ind

            self.peaks = np.array(list(self.peaks)+list(indmax[0]))
            update_plot()

        self.span_remove = SpanSelector(
            self.ax,
            onselect=onselect_remove,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:red"),
            drag_from_anywhere=True,
            button=2
        )

        self.span_add = SpanSelector(
            self.ax,
            onselect=onselect_add,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:green"),
            drag_from_anywhere=True,
            button=3
        )

        # onchange
        self.width_slider_min.on_changed(width_slider_changed)
        # self.width_slider_max.on_changed(width_slider_changed)
        self.m_slider.on_changed(update_background)
        self.bgheight_slider.on_changed(update_background)

        # onclick
        self.openfilebutton.on_clicked(open_file)
        self.savebutton.on_clicked(save_peak)
        self.updatebutton.on_clicked(update)

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

if __name__=='__main__':
    xrdwin = XRD_Peak_search_window()

    xrdwin.ax.legend(frameon=True)
    xrdwin.execute()
    
    """ def update(val):
        if type(val)==float or type(val)==int:
            xrdwin.ax.clear()
            xrdwin.ax.set_title(xrdwin.spectrum_filename)
            p = []

            # xrdwin.set_spectrum(xrdwin.spectrum_filename)
            # xrdwin.set_background()
            
            xrdwin.normalize(xrdwin.net_spectrum)
            xrdwin.background = snip(convolve(xrdwin.spectrum, n=35, std=3), m=xrdwin.m_slider.val)
            xrdwin.background = xrdwin.background + xrdwin.bgheight_slider.val
            xrdwin.net_spectrum = xrdwin.spectrum - xrdwin.background
            xrdwin.net_spectrum[xrdwin.net_spectrum<0] = 0
            xrdwin.net_spectrum = np.convolve(xrdwin.net_spectrum, xrdwin.mask, 'same')
            xrdwin.max_val= np.max(xrdwin.spectrum)
            xrdwin.net_spectrum = xrdwin.normalize(xrdwin.net_spectrum)

            xrdwin.ax.plot(xrdwin.channels, xrdwin.spectrum, label='raw', marker='.', lw=2)
            xrdwin.ax.plot(xrdwin.channels, xrdwin.background, label='background', lw=2)
            xrdwin.ax.plot(xrdwin.channels, xrdwin.net_spectrum, label='net spectrum', marker='.', lw=2)

            xrdwin.ax.set_xlabel('angles')
            xrdwin.ax.set_ylabel('intensity')

            xrdwin.ax.set_xlim(xrdwin.channels[0],xrdwin.channels[-1])
            xrdwin.width_list = np.arange(xrdwin.width_slider_min.val, xrdwin.width_slider_max.val, 0.1)
            for width in xrdwin.width_list:
                peaks = sci_sig.find_peaks(np.log(xrdwin.net_spectrum), width=width, height=2)
                p += list(peaks[0])

            xrdwin.peaks = list(dict.fromkeys(p))
            xrdwin.ax.vlines(xrdwin.channels[xrdwin.peaks], 0, xrdwin.max_val, 'r', label='peak position')
            xrdwin.ax.legend(frameon=True)
            xrdwin.fig.canvas.draw_idle() """
    