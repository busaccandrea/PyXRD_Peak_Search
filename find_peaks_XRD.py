import scipy.signal as sci_sig
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, SpanSelector, CheckButtons
from tkinter import Tk    # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from numpy import minimum,fft,pad
from scipy.signal import windows
from os import path, makedirs

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

        self.linespec, = self.ax.plot(self.channels, self.spectrum, label='spectrum', marker='.')
        self.linebg,   = self.ax.plot(self.channels, self.background, label='background')
        self.linenet,  = self.ax.plot(self.channels, self.net_spectrum, label='net spectrum', marker='.')
        self.vlines = self.ax.vlines(self.channels[self.peaks[0]], 0, 1000, 'r', label='peak position')#self.channels[self.peaks[0]], 0, self.max_val, 'r', label='peak position')
        self.init_widgets()
        self.update_plot()

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
        self.output_dir = './data/'
        self.spectrum = np.array(spectrum)
        self.channels = np.array(channels)
        self.width_list = np.linspace(self.width_default, 50, 10)
        self.spectrum = self.normalize(self.spectrum)
        self.max_val = np.max(self.spectrum)
        
        self.background = snip(convolve(self.spectrum, n=35, std=3), m=16)
        self.net_spectrum = self.spectrum - self.background
        self.net_spectrum[self.net_spectrum<0]=0

        self.smoothing_std_default = 3
        self.smoothing_std = self.smoothing_std_default
        self.net_spectrum = convolve(self.net_spectrum, n=35, std=3)

        self.net_spectrum = self.normalize(self.net_spectrum)

        # find peaks with default settings.
        self.peaks = sci_sig.find_peaks(np.log(self.net_spectrum), width=10, height=2)[0]

        # once channels is initialized we can set lims on x axis
        self.ax.set_xlim(self.channels[0],self.channels[-1])

    def open_new_file(self):
        Tk().withdraw()
        filename = askopenfilename(filetypes=(('dat files', '*.dat'),('All files', '*.*')), initialdir='./')
        return filename

    def init_widgets(self):
        # SLIDERS
        self.axwidth_min = plt.axes([0.25, 0.20, 0.215, 0.03])
        self.width_slider_min = Slider(
            ax=self. axwidth_min,
            label='width min',
            valmin=0,
            valmax=30,
            valstep = 0.1,
            valinit=10
        )

        self.axsmoothing = plt.axes([0.25, 0.1, 0.215, 0.03])
        self.slider_smoothing = Slider(
            ax=self.axsmoothing,
            label='smoothing',
            valmin=0,
            valmax=10,
            valstep = 0.5,
            valinit=self.smoothing_std
        )

        self.axbgheight = plt.axes([0.25, 0.15, 0.65, 0.03])
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

        # BUTTONS
        self.openfileax = plt.axes([0.16, 0.855, 0.06, 0.04])
        self.openfilebutton = Button(self.openfileax, 'Open file', hovercolor='0.975')

        self.updateax = plt.axes([0.16, 0.805, 0.06, 0.04])
        self.updatebutton = Button(self.updateax, 'Update plot', hovercolor='0.975')

        self.restore_default_view_ax = plt.axes([0.16, 0.755, 0.06, 0.04])
        self.restore_default_view_button = Button(self.restore_default_view_ax, 'Restore view', hovercolor='0.975')
        
        self.saveax = plt.axes([0.8, 0.025, 0.04, 0.04])
        self.savebutton = Button(self.saveax, 'Save peaks', hovercolor='0.975')

        # CHECKBOXES
        self.update_lines()

        self.checkax = plt.axes([0.80, 0.655, 0.1, 0.08])
        self.check = CheckButtons(self.checkax, self.labels, self.visibility)
    
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
        if self.smoothing_std!=0:    
            self.net_spectrum = convolve(self.net_spectrum, n=35, std=self.smoothing_std)
        self.max_val= np.max(self.net_spectrum)
        self.net_spectrum = self.normalize(self.net_spectrum)

    def set_peaks(self):
        sci_sig.find_peaks(np.log(self.net_spectrum), width=self.width_default, height=2)

    def update_peaks(self):
        p = []
        for width in self.width_list:
            peaks = sci_sig.find_peaks(np.log(self.net_spectrum), width=width, height=2)
            p += list(peaks[0])
        self.peaks = np.array(list(dict.fromkeys(p)))

    def normalize(self, arr):
        arr = 1000 * arr / np.max(arr)
        return arr

    def update_lines(self):
        self.lines = [self.linespec, self.linebg, self.linenet]
        self.labels = [str(line.get_label()) for line in self.lines]
        self.visibility = [line.get_visible() for line in self.lines]

    def update_plot(self):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.clear()
            self.ax.set_title(path.basename(self.spectrum_filename))

            self.linespec, = self.ax.plot(self.channels, self.spectrum, label='spectrum', marker='.', lw=2)
            self.linebg, = self.ax.plot(self.channels, self.background, label='background', lw=2)
            self.linenet, = self.ax.plot(self.channels, self.net_spectrum, label='net spectrum', marker='.', lw=2)

            self.linespec.set_visible(self.visibility[0])
            self.linebg.set_visible(self.visibility[1])
            self.linenet.set_visible(self.visibility[2])
            
            self.ax.vlines(self.channels[self.peaks], 0, 1000, 'r', label='peak position')
            self.ax.legend(frameon=True)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.fig.canvas.draw_idle()

    def execute(self):
        
        def update_background(val):
            self.set_background()
            self.update_peaks()

            self.update_plot()

        def width_slider_changed(val):
            # I have to update the widthlist
            # self.width_list = np.arange(self.width_slider_min.val, self.width_slider_max.val, 0.1)
            self.width_list = np.arange(self.width_slider_min.val, 50, 0.1)
            # I have to update peaks
            self.update_peaks()
            self.update_plot()

        def open_file(event):
            Tk().withdraw()
            filename = askopenfilename(filetypes=(('dat files', '*.dat'),('All files', '*.*')))
            self.set_spectrum(filename)
            self.set_background()
            self.update_peaks()
            self.update_plot()

        def save_peak(event):
            base, outputfile = path.split(self.spectrum_filename)
            outputfile, frmt = outputfile.split('.')
            outputfile = self.output_dir + outputfile

            # savefig
            extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(outputfile+'.png', bbox_inches=extent.expanded(1.1, 1.2))
            self.net_spectrum = self.normalize(self.net_spectrum)
            # save peaks: in each row of a file write position and intensity of a peak
            with open(outputfile+'.txt', 'w') as outfile:
                for i, channel in enumerate(self.peaks):
                    outfile.write(str(self.channels[channel]) + ' ' + format(self.net_spectrum[channel], '.2f') +'\n')
            with open(outputfile+'_netSpectrum.txt', 'w') as netoutfile:
                for c, channel in enumerate(self.channels):
                    netoutfile.write(str(channel) + ' ' + format(self.net_spectrum[c], '.2f') +'\n')
                    
        def update(event):
            self.update_plot()

        def onselect_remove(xmin, xmax):
            
            def theta_to_channels(theta):
                positions = []
                for t in theta:
                    positions += [np.where(self.channels == t)[0][0]]
                return positions

            print('AVAILABLE peaks:', self.peaks)
            # take all peaks between min and max and remove them
            self.peaks = np.array(self.peaks)
            peak_energy_pos = self.channels[self.peaks]

            peaks2remove = peak_energy_pos[peak_energy_pos>=xmin]
            peaks2remove = peaks2remove[peaks2remove<=xmax]


            peakpos = theta_to_channels(peaks2remove)
            print('SELECTED peak/s (theta):', peaks2remove)
            print('SELECTED peak/s (index):', peakpos)
            for p in peakpos:
                idx = np.where(self.peaks == p)[0][0]
                self.peaks = np.delete(self.peaks, idx)
            
            print('REMAINING peaks:', self.peaks)
            
            self.update_plot()

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

            self.peaks = np.sort(np.array(list(self.peaks)+list(indmax[0])))
            print(self.peaks)
            self.update_plot()

        def smoothing_changed(val):
            self.smoothing_std = self.slider_smoothing.val
            self.set_background()
            self.update_peaks()
            self.update_plot()


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

        def func(label):
            index = self.labels.index(label)
            self.update_lines()
            self.lines[index].set_visible(not self.lines[index].get_visible())
            self.update_lines()
            plt.draw()
            
        
        # onchange
        self.width_slider_min.on_changed(width_slider_changed)
        self.slider_smoothing.on_changed(smoothing_changed)
        self.m_slider.on_changed(update_background)
        self.bgheight_slider.on_changed(update_background)

        # onclick
        self.openfilebutton.on_clicked(open_file)
        self.savebutton.on_clicked(save_peak)
        self.updatebutton.on_clicked(update)
        self.restore_default_view_button.on_clicked(update)
        self.check.on_clicked(func)

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

if __name__=='__main__':
    if not path.isdir('./data/'): makedirs('./data/')
    xrdwin = XRD_Peak_search_window()

    xrdwin.ax.legend(frameon=True)
    xrdwin.execute()