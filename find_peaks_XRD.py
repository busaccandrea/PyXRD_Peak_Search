from turtle import onkeypress
import scipy.signal as sci_sig
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from matplotlib.widgets import Slider, Button, SpanSelector, CheckButtons, TextBox
from tkinter import Tk    # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from numpy import minimum,fft,pad
from scipy.signal import windows
from os import path, makedirs
import yaml
from tkinter.messagebox import askyesno, showinfo
from tkinter import filedialog
from glob import glob
from shutil import copyfile
import tkinter as tk

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

def read_cif_file(ciffilename):
    with open(ciffilename, 'r') as ciffile:
        lines = ciffile.readlines()
        cif_options = {
            'chemical_formula_sum': '',
            'chemical_name_mineral': '',
            'chemical_name_common': '',
            'cell_measurement_wavelength': '',    
            'theta': [],
            'intensity': []
        }
        peak_append = False
        for line in lines:
            if 'chemical_formula_sum' in line:
                cif_options['chemical_formula_sum'] = line.split('  ')[-1]
            elif 'chemical_name_mineral' in line:
                cif_options['chemical_name_mineral'] = line.split('  ')[-1]
            elif 'chemical_name_common' in line:
                cif_options['chemical_name_common'] = line.split('  ')[-1]
            elif 'cell_measurement_wavelength' in line:
                cif_options['cell_measurement_wavelength'] = line.split('  ')[-1]
            elif 'pd_peak_intensity' in line:
                peak_append=True
                continue
            if peak_append:
                splitted = line.split()
                if '#' in splitted[0]: 
                    peak_append=False
                    continue
                cif_options['theta'] += [float(splitted[0])]
                cif_options['intensity'] += [float(splitted[1])]
        
        cif_options['theta'] =  d_to_thetas(cif_options['theta'])
        # print('thetas', cif_options['theta'])
        return cif_options

def d_to_thetas(thetas):
    d = np.array(thetas)
    d = 1.541874/(2*d)
    return np.arcsin(d)*(360/np.pi)

def thetas_to_d(thetas):
    g = np.sin(np.pi * thetas / 360)
    d = 1.541874 / (2 * g)
    # print(thetas)
    # print(d)
    return d

def convert_database():
    print('Select folder where cif files are located')
    Tk().withdraw()
    source_folder = filedialog.askdirectory(title='Select folder where cif files are located')
    destination_folder = './phases/'
    if not path.isdir(destination_folder): makedirs(destination_folder)
    cifs = glob(source_folder+'/*.cif')
    if len(cifs) == 0: print('No \'.cif\' files found!')
    for cif in cifs:
        cif_options = read_cif_file(cif)
        _, filename = path.split(cif)
        s = cif_options['chemical_formula_sum']
        s = s.replace(' ', '_')
        s = s.replace('\'', '')
        s = s.replace('\n', '')
        new_filename = destination_folder+s+'____'+filename
        copyfile(cif, new_filename)

def _tkmakeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.NO, fill=tk.X)
        entries.append((field, ent))
        print('type ent',type(ent))
    return entries

class XRD_Peak_search_window:
    def __init__(self) -> None:

        self.filter_size = 20
        self.min_width = 2
        self.max_width = 50
        self.width_default = 10

        self.fig, self.ax = plt.subplots()
        self.x_default_lims = [0,0]
        self.y_default_lims = [0,0]
        self.ax.set_xlabel('angles')
        self.ax.set_ylabel('intensity')

        plt.subplots_adjust(left=0.35, bottom=0.35, right=0.95)
        
        self.init_spectrum()

        self.linespec, = self.ax.plot(self.channels, self.spectrum, label='spectrum', marker='.')
        self.linebg,   = self.ax.plot(self.channels, self.background, label='background')
        self.linenet,  = self.ax.plot(self.channels, self.net_spectrum, label='net spectrum', marker='.')
        self.vlines = self.ax.vlines(self.channels[self.peaks[0]], 0, 1000, colors='k', linestyles='dashed', label='peak position')#self.channels[self.peaks[0]], 0, self.max_val, 'r', label='peak position')
        self.cif_options = None
        self.cif_files = []
        self.current_cif_index = 0
        self.current_ciffile = None
        self.asterisks = [] 
        self.cif_table = None

        self.init_widgets()
        self.update_plot()

    def init_spectrum(self):
        filename = self.open_new_file()

        with open(filename,'r') as f:
            lines = f.readlines()

        channels, spectrum = [],[]
        for line in lines: 
            line = line.replace('\n', '')
            ch, intensity = line.split(' ')
            spectrum += [float(intensity)]
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

        self.smoothing_std_default = 0
        self.smoothing_std = self.smoothing_std_default
        self.net_spectrum = convolve(self.net_spectrum, n=35, std=3)

        self.net_spectrum = self.normalize(self.net_spectrum)

        # find peaks with default settings.
        self.peaks = sci_sig.find_peaks(np.log(self.net_spectrum), width=10, height=2)[0]
        self.show_peaks = True
        # once channels is initialized we can set lims on x axis
        self.x_default_lims=[self.channels[0], self.channels[-1]]
        self.y_default_lims=[-50, 1050]
        self.ax.set_xlim(self.x_default_lims)
        self.ax.set_ylim(self.y_default_lims)

    def open_new_file(self, initialdir='./data/'):
        Tk().withdraw()
        filename = askopenfilename(filetypes=(('dat files', '*.dat'),('All files', '*.*')), initialdir=initialdir)
        return filename

    def init_widgets(self):
        # SLIDERS
        self.axwidth_min = plt.axes([0.35, 0.20, 0.215, 0.03])
        self.width_slider = Slider(
            ax=self. axwidth_min,
            label='width min',
            valmin=0,
            valmax=30,
            valstep = 0.1,
            valinit=10
        )
        self.axwidth_addbutton = plt.axes([0.59, 0.22, 0.015, 0.01])
        self.width_slider_min_addbutton = Button(self.axwidth_addbutton, '+', hovercolor='0.975')
        self.axwidth_subbutton = plt.axes([0.59, 0.205, 0.015, 0.01])
        self.width_slider_min_subbutton = Button(self.axwidth_subbutton, '-', hovercolor='0.975')

        self.axsmoothing = plt.axes([0.35, 0.1, 0.215, 0.03])
        self.smoothing_slider = Slider(
            ax=self.axsmoothing,
            label='smoothing',
            valmin=0,
            valmax=10,
            valstep = 0.5,
            valinit=0
        )
        self.axsmoothing_addbutton = plt.axes([0.59, 0.12, 0.015, 0.01])
        self.smoothing_slider_min_addbutton = Button(self.axsmoothing_addbutton, '+', hovercolor='0.975')
        self.axsmoothing_subbutton = plt.axes([0.59, 0.105, 0.015, 0.01])
        self.smoothing_slider_min_subbutton = Button(self.axsmoothing_subbutton, '-', hovercolor='0.975')

        self.axbgheight = plt.axes([0.35, 0.15, 0.55, 0.03])
        self.bgheight_slider = Slider(
        ax=self.axbgheight,
        label='background height',
        valmin=0,
        valmax=400,
        valstep = 1,
        valinit=0
        )
        self.axbheight_addbutton = plt.axes([0.93, 0.17, 0.015, 0.01])
        self.bheight_slider_min_addbutton = Button(self.axbheight_addbutton, '+', hovercolor='0.975')
        self.axbheight_subbutton = plt.axes([0.93, 0.155, 0.015, 0.01])
        self.bheight_slider_min_subbutton = Button(self.axbheight_subbutton, '-', hovercolor='0.975')
        
        self.axsnip = plt.axes([0.684, 0.20, 0.215, 0.03])
        self.snip_slider = Slider(
        ax=self.axsnip,
        label='snip width',
        valmin=10,
        valmax=50,
        valstep = 1,
        valinit=16
        )
        self.axsnip_addbutton = plt.axes([0.93, 0.22, 0.015, 0.01])
        self.snip_slider_min_addbutton = Button(self.axsnip_addbutton, '+', hovercolor='0.975')
        self.axsnip_subbutton = plt.axes([0.93, 0.205, 0.015, 0.01])
        self.snip_slider_min_subbutton = Button(self.axsnip_subbutton, '-', hovercolor='0.975')

        self.axdeltatheta = plt.axes([0.684, 0.1, 0.215, 0.03])
        self.deltatheta_slider = Slider(
        ax=self.axdeltatheta,
        label=r'$\Delta$ theta',
        valmin=0.01,
        valmax=0.5,
        valstep = 0.01,
        valinit=0.1
        )
        self.axdeltatheta_addbutton = plt.axes([0.93, 0.12, 0.015, 0.01])
        self.deltatheta_slider_min_addbutton = Button(self.axdeltatheta_addbutton, '+', hovercolor='0.975')
        self.axdeltatheta_subbutton = plt.axes([0.93, 0.105, 0.015, 0.01])
        self.deltatheta_slider_min_subbutton = Button(self.axdeltatheta_subbutton, '-', hovercolor='0.975')

        # BUTTONS
        self.openfileax = plt.axes([0.26, 0.855, 0.06, 0.04])
        self.openfilebutton = Button(self.openfileax, 'Open file', hovercolor='0.975')

        self.updateax = plt.axes([0.26, 0.805, 0.06, 0.04])
        self.updatebutton = Button(self.updateax, 'Update plot', hovercolor='0.975')

        self.loadconfigax = plt.axes([0.26, 0.605, 0.06, 0.04])
        self.loadconfigbutton = Button(self.loadconfigax, 'Load config', hovercolor='0.975')

        self.cifdescriptionax = plt.axes([0.0, 0.0, 0.9, 0.9])
        self.cifdescriptionax.set_title('description of cif loaded')
        self.cifdescriptiontext = 'No .cif file loaded.'
        self.cifdescriptionax.text(0,0.8,self.cifdescriptiontext)

        self.textboxax = self.fig.add_axes([0.001, 0.705, 0.2, 0.03])
        self.textbox = TextBox(self.textboxax, '')

        self.ciflistax = plt.axes([0.001, 0.001, 0.2, 0.7])
        self.ciflistax.set_title('description of cif found')
        self.ciflisttext = ''
        self.ciflistax.text(0,0.8,self.ciflisttext)

        self.loadcifax = plt.axes([0.26, 0.555, 0.06, 0.04])
        self.loadcifbutton = Button(self.loadcifax, 'Load cif DB', hovercolor='0.975')

        self.clearcifax = plt.axes([0.22, 0.555, 0.02, 0.04])
        self.clearcifbutton = Button(self.clearcifax, 'X', color='red', hovercolor='0.975')

        self.restore_default_view_ax = plt.axes([0.26, 0.755, 0.06, 0.04])
        self.restore_default_view_button = Button(self.restore_default_view_ax, 'Restore view', hovercolor='0.975')
        
        self.saveax = plt.axes([0.8, 0.025, 0.06, 0.04])
        self.savebutton = Button(self.saveax, 'Save peaks', hovercolor='0.975')

        # CHECKBOXES
        self.update_lines()
        self.checkax = plt.axes([0.22, 0.355, 0.1, 0.1])
        self.checkax.text(0,1.1, 'show/hide lines box')
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
        self.background = snip(convolve(self.spectrum, n=35, std=3), m=self.snip_slider.val)
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
        self.lines = [self.linespec, self.linebg, self.linenet]#, self.vlines]
        self.labels = [str(line.get_label()) for line in self.lines]
        self.visibility = [line.get_visible() for line in self.lines]
        print('on update lines', self.visibility)

    def update_plot(self):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.clear()
            self.ax.set_title(path.basename(self.spectrum_filename))
            self.set_background()
            self.linespec, = self.ax.plot(self.channels, self.spectrum, color='#333333', alpha=0.8, label='spectrum')
            self.linebg, = self.ax.plot(self.channels, self.background, color='#f7a072', label='background', lw=3)
            self.linenet, = self.ax.plot(self.channels, self.net_spectrum, color='#98CE00', label='net spectrum', marker='.', ms=4)

            self.cifdescriptionax.remove()
            del self.cifdescriptionax
            self.cifdescriptionax = plt.axes([0.0, 0.79, 0.25, 0.2])
            self.cifdescriptionax.text(0.01, 0.97, self.cifdescriptiontext, ha='left', va='top', fontweight='bold')
            self.cifdescriptionax.set_xticklabels([])
            self.cifdescriptionax.set_yticklabels([])
            
            self.ciflistax.remove()
            del self.ciflistax
            self.ciflistax = plt.axes([0.001, 0.001, 0.2, 0.7])
            self.ciflistax.set_xticklabels([])
            self.ciflistax.set_yticklabels([])

            if self.cif_options != None:
                self.ax.vlines(self.cif_options['theta'], 0, self.cif_options['intensity'], colors='r', linestyles='solid', label='peaks from db', lw=2)

                self.asterisks = [] 
                celltext = []
                for dbpk, dbpeak in enumerate(self.cif_options['theta']):
                    for p, peak in enumerate(self.channels[self.peaks]):
                        if np.abs(dbpeak - peak) < self.deltatheta_slider.val: 
                            self.asterisks += [peak]
                            celltext.append([format(peak, '.2f'),format(dbpeak, '.2f')])

                self.ax.plot(self.asterisks, np.zeros(len(self.asterisks))-20, 'k', linewidth=0, marker='*', ms=10)
                self.cifdescriptionax.set_facecolor((128/255, 241/255, 132/255))

                missing_peaks = self.cif_options['theta'][self.cif_options['theta']>self.x_default_lims[0]]
                missing_peaks = missing_peaks[missing_peaks<self.x_default_lims[1]]
                
                celltext.append(['# missing peaks', str(missing_peaks.size - len(self.asterisks))])
                self.ciflistax.table(cellText=celltext, colLabels=['Exp', 'cif'], loc='upper center', cellLoc='center')

            else: 
                self.cifdescriptionax.set_facecolor((0.9, 0.9, 0.9))
            
            self.ax.vlines(self.channels[self.peaks], 0, 1000, colors='#248794', alpha=0.8, linewidth=2, linestyles='dashed', label='peak position')

            self.linespec.set_visible(self.visibility[0])
            self.linebg.set_visible(self.visibility[1])
            self.linenet.set_visible(self.visibility[2])

            self.ax.legend(frameon=True)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.grid(True)
            self.fig.canvas.draw_idle()

    def execute(self):
        
        def update_background(val):
            self.set_background()
            self.update_peaks()

            self.update_plot()

        def width_slider_changed(val):
            self.width_list = np.arange(self.width_slider.val, 50, 0.1)
            self.update_peaks()
            self.update_plot()

        def deltatheta_slider_changed(val):
            # print('deltatheta', self.cif_options['theta'])
            self.update_plot()

        def open_file(event):
            Tk().withdraw()
            filename = askopenfilename(filetypes=(('dat files', '*.dat'),('All files', '*.*')))
            self.set_spectrum(filename)
            self.set_background()
            self.update_peaks()
            self.update_plot()

        def save_peak(event):
            c = askyesno(title='Select yes or no.', message='Do you want to save the output with .cif structure?\nTheta will be replaced by d.')
            base, outputfile = path.split(self.spectrum_filename)
            outputfile, frmt = outputfile.split('.')
            outputfile = self.output_dir + outputfile
            if c:
                win = Tk()
                
                peaksfile = outputfile +'.cif'
                fields = ('_formula_sum', '_name_mineral', '_name_common')
                ents = _tkmakeform(win, fields)
                b2 = tk.Button(win, text='Save&Close', command=win.quit)
                b2.pack(side=tk.LEFT, padx=5, pady=5)
                
                win.mainloop()  
                
                formula = '\''+ ents[0][1].get() + '\''
                namecommon = '\''+ ents[1][1].get() + '\''
                namemineral = '\''+ ents[2][1].get() + '\''

                win.destroy()
            else: 
                peaksfile = outputfile +'.txt'
                formula = ''
                namecommon = ''
                namemineral = ''

            # savefig
            extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(outputfile+'.png', bbox_inches=extent.expanded(1.1, 1.2))
            self.net_spectrum = self.normalize(self.net_spectrum)

            with open(peaksfile, 'w') as outfile:
                # write loop_, _pd_peak_d_spacing, _pd_peak_intensity
                outfile.write('_chemical_formula_sum  ' + formula +'\n')
                outfile.write('_chemical_name_common  ' + namecommon +'\n')
                outfile.write('_chemical_name_mineral  ' + namemineral +'\n')
                outfile.write('loop_\n')
                outfile.write('_pd_peak_d_spacing\n')
                outfile.write('_pd_peak_intensity\n')

                if c:
                    #save cif
                    d = thetas_to_d(self.channels[self.peaks])
                    print('writing d',d)
                    for i, d_ in enumerate(d):
                        intensity = self.net_spectrum[self.peaks[i]]
                        intensity = format(intensity, '.2f')
                        outfile.write('     ' + str(format(d[i], '.6f')) + f'{str(intensity):>14}' +'\n')
                    
                # save peaks: in each row of a file write position and intensity of a peak
                else:
                    print('writing', self.channels[self.peaks])
                    for i, channel in enumerate(self.peaks):
                        intensity = str(format(self.net_spectrum[channel], '.2f'))
                        outfile.write('     ' + str(self.channels[channel]) + ' ' + f'{intensity:>14}' +'\n')

            with open(outputfile+'_netSpectrum.txt', 'w') as netoutfile:
                for c, channel in enumerate(self.channels):
                    netoutfile.write(str(channel) + ' ' + format(self.net_spectrum[c], '.2f') +'\n')
            
            options = {
                'width': self.width_slider.val,
                'smoothing': self.smoothing_slider.val,
                'background_height' : self.bgheight_slider.val,
                'snip' : self.snip_slider.val}
            with open(outputfile+'.yaml', 'w') as yamlfile: yaml.dump(options, yamlfile)

            showinfo(title='Info', message='Saved!')

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
            # print(self.peaks)
            self.update_plot()

        def smoothing_changed(val):
            self.smoothing_std = self.smoothing_slider.val
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

        def show_hide_lines(label):
            self.update_lines()
            index = self.labels.index(label)
            # print(self.lines[index].get_visible())
            self.lines[index].set_visible(not self.lines[index].get_visible())
            # print(self.lines[index].get_visible())
            self.update_lines()
            plt.draw()
            
        def load_config(loadconfigevent):
            Tk().withdraw()
            yamlfilename = askopenfilename(filetypes=(('yaml files', '*.yaml'),('All files', '*.*')))
            options = None
            with open(yamlfilename) as yamlfile: options = yaml.full_load(yamlfile)
            
            self.bgheight_slider.set_val(options['background_height'])
            self.width_slider.set_val(options['width'])
            self.smoothing_slider.set_val(options['smoothing'])
            self.snip_slider.set_val(options['snip'])
            self.update_plot()

        def load_cif_db(loadcifevent):
            Tk().withdraw()
            ciffilename = askopenfilename(filetypes=(('cif files', '*.cif'),('All files', '*.*')), initialdir='./phases/')
            self.cif_options = read_cif_file(ciffilename=ciffilename)
            self.cifdescriptiontext = 'chemical formula:\n' + self.cif_options['chemical_formula_sum']+\
                '\nchemical name mineral:\n'+self.cif_options['chemical_name_mineral']+\
                    '\nchemical name common:\n'+self.cif_options['chemical_name_common']
            self.update_plot()

        def clear_cif_db(clearcifevent):
            self.cif_options = None
            self.cifdescriptiontext = 'No .cif file loaded.'
            self.cif_files = []
            self.current_cif_index = 0
            self.ciflisttext = ''
            self.textbox.set_val('')

            self.update_plot()

        def restore_view(event):
            self.ax.set_xlim(self.x_default_lims)
            self.ax.set_ylim(-50, 1050)
            self.update_plot()
        
        # onchange
        self.width_slider.on_changed(width_slider_changed)
        self.smoothing_slider.on_changed(smoothing_changed)
        self.snip_slider.on_changed(update_background)
        self.bgheight_slider.on_changed(update_background)
        self.deltatheta_slider.on_changed(deltatheta_slider_changed)

        # onclick
        self.openfilebutton.on_clicked(open_file)
        self.savebutton.on_clicked(save_peak)
        self.updatebutton.on_clicked(update)
        self.restore_default_view_button.on_clicked(restore_view)
        self.check.on_clicked(show_hide_lines)
        self.loadconfigbutton.on_clicked(load_config)
        self.loadcifbutton.on_clicked(load_cif_db)
        self.clearcifbutton.on_clicked(clear_cif_db)
        
        def smoothing_add(addevent): 
            if self.smoothing_slider.val < self.smoothing_slider.valmax:self.smoothing_slider.set_val(self.smoothing_slider.val + self.smoothing_slider.valstep)
        def smoothing_sub(subevent): 
            if self.smoothing_slider.val > self.smoothing_slider.valmin:self.smoothing_slider.set_val(self.smoothing_slider.val - self.smoothing_slider.valstep)
        def bheight_add(addevent): 
            if self.bgheight_slider.val < self.bgheight_slider.valmax:self.bgheight_slider.set_val(self.bgheight_slider.val + self.bgheight_slider.valstep)
        def bheight_sub(subevent): 
            if self.bgheight_slider.val > self.bgheight_slider.valmin:self.bgheight_slider.set_val(self.bgheight_slider.val - self.bgheight_slider.valstep)
        def deltatheta_add(addevent): 
            if self.deltatheta_slider.val < self.deltatheta_slider.valmax:self.deltatheta_slider.set_val(self.deltatheta_slider.val + self.deltatheta_slider.valstep)
        def deltatheta_sub(subevent): 
            if self.deltatheta_slider.val > self.deltatheta_slider.valmin:self.deltatheta_slider.set_val(self.deltatheta_slider.val - self.deltatheta_slider.valstep)
        def snip_add(addevent): 
            if self.snip_slider.val < self.snip_slider.valmax:self.snip_slider.set_val(self.snip_slider.val + self.snip_slider.valstep)
        def snip_sub(subevent): 
            if self.snip_slider.val > self.snip_slider.valmin:self.snip_slider.set_val(self.snip_slider.val - self.snip_slider.valstep)
        def width_add(addevent): 
            if self.width_slider.val < self.width_slider.valmax:self.width_slider.set_val(self.width_slider.val + self.width_slider.valstep)
        def width_sub(subevent): 
            if self.width_slider.val > self.width_slider.valmin:self.width_slider.set_val(self.width_slider.val - self.width_slider.valstep)

        self.smoothing_slider_min_addbutton.on_clicked(smoothing_add)
        self.smoothing_slider_min_subbutton.on_clicked(smoothing_sub)
        self.bheight_slider_min_addbutton.on_clicked(bheight_add)
        self.bheight_slider_min_subbutton.on_clicked(bheight_sub)
        self.deltatheta_slider_min_addbutton.on_clicked(deltatheta_add)
        self.deltatheta_slider_min_subbutton.on_clicked(deltatheta_sub)
        self.snip_slider_min_addbutton.on_clicked(snip_add)
        self.snip_slider_min_subbutton.on_clicked(snip_sub)
        self.width_slider_min_addbutton.on_clicked(width_add)
        self.width_slider_min_subbutton.on_clicked(width_sub)

        def submit(submittext):
            self.current_cif_index = 0
            self.cif_files = []

            if submittext != '':
                filelist = []
                
                if ',' in submittext or ' ' in submittext:
                    if ',' in submittext:
                        submittext = submittext.replace(' ', '')
                        elements = submittext.split(',')
                        
                    else: elements = submittext.split(' ')
                    
                    filelist = glob('./phases/*'+elements[0]+'*')
                    # print('\n\nfilelist iniziale!!!!!', len(filelist),'\n',filelist)
                    for element in elements:
                        filelist_ = filelist.copy()
                        for f in filelist_:
                            if not (element in path.basename(f)): filelist.remove(f); print('file removed because', element, 'not in',  path.basename(f), len (filelist))
                            
                            else: print('file keeped because', element, 'in', path.basename(f))

                    # print('\n\nfilelist finale!!!!!', len(filelist),'\n',filelist, '\n\n\n')

                    

                else: filelist = glob('./phases/*'+submittext+'*')

                fl = ''
                for f in filelist:
                    filename = path.basename(f)
                    fl += filename+'\n'
                    self.cif_files.append(filename)

                self.cif_options = read_cif_file(ciffilename='./phases/'+self.cif_files[self.current_cif_index])
                self.cifdescriptiontext = 'number of files found:   ' + str(len(self.cif_files))+'\n\nchemical formula:\n' + self.cif_options['chemical_formula_sum']+\
                    '\nchemical name mineral:\n'+self.cif_options['chemical_name_mineral']+\
                        '\nchemical name common:\n'+self.cif_options['chemical_name_common']

            else: 
                self.cif_options = None
                self.cifdescriptiontext = 'No .cif file loaded.'

            self.update_plot()
                
        self.textbox.on_submit(submit)

        def onkeypress(key):
            if key.key == 'left' or key.key == 'right':
                if key.key == 'right':
                    print('pressed right key')
                    if self.current_cif_index<len(self.cif_files)-1: 
                        self.current_cif_index += 1 

                if key.key == 'left':
                    print('pressed left key')
                    if self.current_cif_index>0:
                        self.current_cif_index -= 1 
                
                self.cif_options = read_cif_file(ciffilename='./phases/'+self.cif_files[self.current_cif_index])
                self.cifdescriptiontext = 'number of files found:   ' + str(len(self.cif_files))+'\n\nchemical formula:\n' + self.cif_options['chemical_formula_sum']+\
                    '\nchemical name mineral:\n'+self.cif_options['chemical_name_mineral']+\
                        '\nchemical name common:\n'+self.cif_options['chemical_name_common']

                self.update_plot()

        self.fig.canvas.mpl_connect('key_press_event', onkeypress)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

if __name__=='__main__':
    if not path.isdir('./data/'): makedirs('./data/')

    if not glob('./phases/*'):
        root = tk.Tk()
        c = askyesno(title='select yes or no.', message='Do you want to import a new .cif database? \nThe output will be saved in the ./phases/ folder.\
        \nWARNING! Imported files will be renamed to recognize phases.')
        root.destroy()
        if c: convert_database()

    xrdwin = XRD_Peak_search_window()

    xrdwin.ax.legend(frameon=True)
    xrdwin.execute()