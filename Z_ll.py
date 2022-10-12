import uproot # for reading .root files
import awkward as ak # to represent nested data in columnar format
import vector # for 4-momentum calculations
import time # to measure time to analyse
import math # for mathematical functions such as square root
import numpy as np # for numerical calculations such as histogramming
import numpy.ma as ma
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import AutoMinorLocator # for minor ticks

import infofile # local file containing cross-sections, sums of weights, dataset IDs

#Lumi, fraction, file path
#lumi = 0.5 # fb-1 # data_A only 
#lumi = 1.9 # fb-1 # data_B only
#lumi = 2.9 # fb-1 # data_C only
#lumi = 4.7 # fb-1 # data_D only
lumi = 10 # fb-1 # data_A,data_B,data_C,data_D

fraction = 0.001 # reduce this is if you want the code to run quicker
                                                                                                                                  
#tuple_path = "Input/2lep/" # local 
tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/2lep/" # web address

#SAMPLES

samples = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D'],
    },

    r'$Z/(gamma)* -> ll$' : { #Z + jets
        'list' : ['Zee','Zmumu', 'Ztautau'],
        'color' : "#eef207" # yellow
    },

    r'$t\bar{t}$' : { #ttbar
        'list' : ['ttbar_lep'],
        'color' : "#6b59d3" # purple
    },

    #r'$Diboson$' : { # WW, ZZ, WZ
     #   'list' : ['ZqqZll','llvv',#'WqqZll','WpqqWmlv','WlvZqq','llll','lllv','lvvv'
      #  ],
       # 'color' : "#1fa125" # green
    #},

     #r'$DY$' : { #DY
      #  'list' : ['Zmumu_PTV0_70_CVetoBVeto','Zmumu_PTV0_70_CFilterBVeto','Zmumu_PTV0_70_BFilter','Zmumu_PTV70_140_CVetoBVeto','Zmumu_PTV70_140_CFilterBVeto','Zmumu_PTV70_140_BFilter','Zmumu_PTV140_280_CVetoBVeto',
      #  'Zmumu_PTV140_280_CFilterBVeto','Zmumu_PTV140_280_BFilter',
      #  'Zmumu_PTV280_500_CVetoBVeto',
      #  'Zmumu_PTV280_500_CFilterBVeto','Zmumu_PTV280_500_BFilter','Zmumu_PTV500_1000',
      #  'Zmumu_PTV1000_E_CMS',
      #  'Zee_PTV0_70_CVetoBVeto','Zee_PTV0_70_CFilterBVeto','Zee_PTV0_70_BFilter','Zee_PTV70_140_CVetoBVeto','Zee_PTV70_140_CFilterBVeto','Zee_PTV70_140_BFilter','Zee_PTV140_280_CVetoBVeto',
      #  'Zee_PTV140_280_CFilterBVeto','Zee_PTV140_280_BFilter',
      #  'Zee_PTV280_500_CVetoBVeto',
      #  'Zee_PTV280_500_CFilterBVeto','Zee_PTV280_500_BFilter','Zee_PTV500_1000',
      #  'Zee_PTV1000_E_CMS',
      #  'Ztautau_PTV0_70_CVetoBVeto','Ztautau_PTV0_70_CFilterBVeto','Ztautau_PTV0_70_BFilter','Ztautau_PTV70_140_CVetoBVeto','Ztautau_PTV70_140_CFilterBVeto','Ztautau_PTV70_140_BFilter','Ztautau_PTV140_280_CVetoBVeto','Ztautau_PTV140_280_CFilterBVeto','Ztautau_PTV140_280_BFilter','Ztautau_PTV280_500_CVetoBVeto','Ztautau_PTV280_500_CFilterBVeto','Ztautau_PTV280_500_BFilter','Ztautau_PTV500_1000','Ztautau_PTV1000_E_CMS'
      #  ],
      #  'color' : "#ff0000" # red
    #},

    #r'$Single top$' : { #Single top
     #   'list' : ['single_top_tchan','single_top_wtchan','single_top_schan'],
     #   'color' : "#00cdff" # light blue
    #},

}

MeV = 0.001
GeV = 1.0

#define function to get data from files
def get_data_from_files():

    data = {} # define empty dictionary to hold awkward arrays
    for s in samples: # loop over samples
        print('Processing '+s+' samples') # print which sample
        frames = [] # define empty list to hold data
        for val in samples[s]['list']: # loop over each file
            if s == 'data': prefix = "Data/" # Data prefix
            else: # MC prefix
                prefix = "MC/mc_"+str(infofile.infos[val]["DSID"])+"."
            fileString = tuple_path+prefix+val+".2lep.root" # file name to open
            temp = read_file(fileString,val) # call the function read_file defined below
            frames.append(temp) # append array returned from read_file to list of awkward arrays
        data[s] = ak.concatenate(frames) # dictionary entry is concatenated awkward arrays
    
    return data # return dictionary of awkward arrays

#define function to calculate weight of MC event
def calc_weight(xsec_weight, events):
    return (
        xsec_weight
        * events.mcWeight
        * events.scaleFactor_PILEUP
        * events.scaleFactor_ELE
        * events.scaleFactor_MUON 
        * events.scaleFactor_LepTRIGGER
    )

#define function to get cross-section weight
def get_xsec_weight(sample):
    info = infofile.infos[sample] # open infofile
    xsec_weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    return xsec_weight # return cross-section weight

#define function to calculate 2-lepton invariant mass
def calc_mll(lep_pt, lep_eta, lep_phi, lep_E):
    # construct awkward 4-vector array
    p4 = vector.awk(ak.zip(dict(pt=lep_pt, eta=lep_eta, phi=lep_phi, E=lep_E)))
    # calculate invariant mass of first 2 leptons
    # [:, i] selects the i-th lepton in each event
    # .M calculates the invariant mass
    #return ((p4[:, 0] + p4[:, 1]).M * MeV > 6600) & ((p4[:, 0] + p4[:, 1]).M * MeV < 11600)
    return (p4[:, 0] + p4[:, 1]).M * MeV

#define function to calculate pt of the Z boson candidate
# .pt calculates the momentum
def Zpt(lep_pts,lep_etas,lep_phis, lep_E):
    p4 = vector.awk(ak.zip(dict(pt=lep_pts, eta=lep_etas, phi=lep_phis, E=lep_E)))
    return (p4[:,0] + p4[:,1]).pt * MeV

#define function to calculate pt of the MET
# .pt calculates the momentum
def met_et(met_pts):
    return met_pts * GeV

#define function to calculate pt of jets
# .pt calculates the momentum
def jetpt(jet_pt,jet_eta,jet_phi,jet_E):
    p4 = vector.awk(ak.zip(dict(pt=jet_pt, eta=jet_eta, phi=jet_phi, E=jet_E)))
    return (p4[:,0] + p4[:,1]).pt * MeV

def jetpt_leading(jet_pt,jet_eta,jet_phi,jet_E):
    p4 = vector.awk(ak.zip(dict(pt=jet_pt, eta=jet_eta, phi=jet_phi, E=jet_E)))
    return (p4[:,0]).pt * MeV

def jetpt_subleading(jet_pt,jet_eta,jet_phi,jet_E):
    p4 = vector.awk(ak.zip(dict(pt=jet_pt, eta=jet_eta, phi=jet_phi, E=jet_E)))
    return (p4[:,1]).pt * MeV

#Changing a cut

#User input to define the study we are going for
type_of_study = int(input('If you want to study the invariant mass enter 1, if you want to study the jets enter 2:'))

if (type_of_study==1):
    ll = int(input('For ee enter 1, for mumu enter 2, for all enter 3:'))
    # cut on lepton charge
    # paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
    def cut_lep_charge(lep_charge):
    # throw away when sum of lepton charges is not equal to 0
    # first lepton in each event is [:, 0], 2nd lepton is [:, 1] etc
        return lep_charge[:, 0] + lep_charge[:, 1] != 0

    # cut on lepton type
    # paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
    if(ll==1):
        def cut_lep_type(lep_type):
            sum_lep_type = lep_type[:, 0] + lep_type[:, 1]
            return (sum_lep_type != 22)
    if(ll==2):
        def cut_lep_type(lep_type):
            sum_lep_type = lep_type[:, 0] + lep_type[:, 1]
            return (sum_lep_type != 26)
    if(ll==3):
        def cut_lep_type(lep_type):
        # for an electron lep_type is 11
        # for a muon lep_type is 13
        # throw away when none of ee, mumu
            sum_lep_type = lep_type[:, 0] + lep_type[:, 1]
            return (sum_lep_type != 22) & (sum_lep_type != 26)

if (type_of_study==2):
    num = int(input('Enter the number of jets you want to study:'))
    # cut on lepton charge
    # paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
    def cut_lep_charge(lep_charge):
    # throw away when sum of lepton charges is not equal to 0
    # first lepton in each event is [:, 0], 2nd lepton is [:, 1] etc
        return lep_charge[:, 0] + lep_charge[:, 1] != 0

    # cut on lepton type
    # paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
    def cut_lep_type(lep_type):
        # for an electron lep_type is 11
        # for a muon lep_type is 13
        # throw away when none of ee, mumu
            sum_lep_type = lep_type[:, 0] + lep_type[:, 1]
            return (sum_lep_type != 22) & (sum_lep_type != 26)
    #cut on good jets, define function to find jets passing some minimum requirements
    # paper: "Jets are accepted if they fulfill the requirements pT > 25 GeV"
    # paper: jets with pT < 60 GeV and |η| < 2.4 are required to satisfy pileup rejection criteria (JVT)
    def cut_jet_pt(jet_pt):
        return jet_pt[:, 0] + jet_pt[:,1] > 25*GeV

    def cut_jet_pt_up(jet_pt):
        return jet_pt[:, 0] + jet_pt[:,1] < 60*GeV

    def cut_jet_n(jet_n):
        return jet_n != num

    def cut_jet_jvt(jet_jvt):
        return jet_jvt[:, 0] > 0.59*GeV

#Applying a cut

def read_file(path,sample):
    start = time.time() # start the clock
    print("\tProcessing: "+sample) # print which sample is being processed
    data_all = [] # define empty list to hold all data for this sample
    
    if (type_of_study==1):
        # open the tree called mini using a context manager (will automatically close files/resources)
        with uproot.open(path + ":mini") as tree:
            numevents = tree.num_entries # number of events
            if 'data' not in sample: xsec_weight = get_xsec_weight(sample) # get cross-section weight
            for data in tree.iterate(['lep_pt','lep_eta','lep_phi','lep_E','lep_charge','lep_type',
                                    'jet_pt','jet_eta','jet_phi','jet_E','jet_n','jet_trueflav', 'jet_jvt','jet_MV2c10',
                                    'met_et','met_phi', 
                                    # add more variables here if you make cuts on them 
                                    'mcWeight','scaleFactor_PILEUP',
                                    'scaleFactor_ELE','scaleFactor_MUON',
                                    'scaleFactor_LepTRIGGER', 'XSection'], # variables to calculate Monte Carlo weight
                                    library="ak", # choose output type as awkward array
                                    entry_stop=numevents*fraction): # process up to numevents*fraction

                nIn = len(data) # number of events in this batch

                if 'data' not in sample: # only do this for Monte Carlo simulation files
                    # multiply all Monte Carlo weights and scale factors together to give total weight
                    data['totalWeight'] = calc_weight(xsec_weight, data)

                # cut on lepton charge using the function cut_lep_charge defined above
                data = data[~cut_lep_charge(data.lep_charge)]

                # cut on lepton type using the function cut_lep_type defined above
                data = data[~cut_lep_type(data.lep_type)]

                # calculation of 2-lepton invariant mass using the function calc_mll defined above
                data['mll'] = calc_mll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_E)

                #calculating pt of the Z boson candidate
                data['Zpt'] = Zpt(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_E)

                #calculating pt of the MET
                data['metpt'] = met_et(data.met_et)
                

                nOut = len(data) # number of events passing cuts in this batch
                data_all.append(data) # append array from this batch
                elapsed = time.time() - start # time taken to process
                print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after
    
        return ak.concatenate(data_all) # return array containing events passing all cuts

    if (type_of_study==2):
        # open the tree called mini using a context manager (will automatically close files/resources)
        with uproot.open(path + ":mini") as tree:
            numevents = tree.num_entries # number of events
            if 'data' not in sample: xsec_weight = get_xsec_weight(sample) # get cross-section weight
            for data in tree.iterate(['lep_pt','lep_eta','lep_phi','lep_E','lep_charge','lep_type',
                                    'jet_pt','jet_eta','jet_phi','jet_E','jet_n','jet_trueflav', 'jet_jvt','jet_MV2c10',
                                    'met_et','met_phi', 
                                    # add more variables here if you make cuts on them 
                                    'mcWeight','scaleFactor_PILEUP',
                                    'scaleFactor_ELE','scaleFactor_MUON',
                                    'scaleFactor_LepTRIGGER', 'XSection'], # variables to calculate Monte Carlo weight
                                    library="ak", # choose output type as awkward array
                                    entry_stop=numevents*fraction): # process up to numevents*fraction

                nIn = len(data) # number of events in this batch

                if 'data' not in sample: # only do this for Monte Carlo simulation files
                    # multiply all Monte Carlo weights and scale factors together to give total weight
                    data['totalWeight'] = calc_weight(xsec_weight, data)

                # cut on lepton charge using the function cut_lep_charge defined above
                data = data[~cut_lep_charge(data.lep_charge)]

                # cut on lepton type using the function cut_lep_type defined above
                data = data[~cut_lep_type(data.lep_type)]

                #cut on number of jets using the function cut_jet_n defined above
                data = data[~cut_jet_n(data.jet_n)]
                        
                #cut jet vertex tagger using the function cut_jet_jvt defined above
                #data = data[~cut_jet_jvt(data.jet_jvt)]

                #calculating pt of the Z boson candidate
                data['Zpt'] = Zpt(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_E)

                #calculating pt of the jets
                data['jetpt_leading'] = jetpt_leading(data.jet_pt,data.jet_eta,data.jet_phi,data.jet_E)
                #data['jetpt_subleading'] = jetpt_subleading(data.jet_pt,data.jet_eta,data.jet_phi,data.jet_E)

                #calculating the difference between Zpt and Jet_pt
                data['JZB'] = data.jetpt_leading - data.Zpt 

                #calculating invariant mass of di-jets
                if (num==2) :
                    data['mjj'] = calc_mll(data.jet_pt,data.jet_eta,data.jet_phi,data.jet_E)


                # cut on good jets using the function cut_goodjets defined above
                #data = data[~cut_jet_pt(data.jetpt)]
                
                data['logos'] = data.jetpt_leading/data.Zpt
                data['logos2'] = data.Zpt/data.jetpt_leading

                nOut = len(data) # number of events passing cuts in this batch
                data_all.append(data) # append array from this batch
                elapsed = time.time() - start # time taken to process
                print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after
    
        return ak.concatenate(data_all) # return array containing events passing all cuts

#prosses strarts here

start = time.time() # time at start of whole processing
data = get_data_from_files() # process all files
elapsed = time.time() - start # time after whole processing
print("Time taken: "+str(round(elapsed,1))+"s") # print total time taken to process every file

#Plotting

if (type_of_study==1):
    #Plotting Mll
    def plot_data(data):

        xmin = 60 * GeV
        xmax = 120 * GeV
        step_size = 1 * GeV

        bin_edges = np.arange(start=xmin, # The interval includes this value
                        stop=xmax+step_size, # The interval doesn't include this value
                        step=step_size ) # Spacing between values
        bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                stop=xmax+step_size/2, # The interval doesn't include this value
                                step=step_size ) # Spacing between values

        data_x,_ = np.histogram(ak.to_numpy(data['data']['mll']), 
                                bins=bin_edges ) # histogram the data
        data_x_errors = np.sqrt( data_x ) # statistical error on the data


        mc_x = [] # define list to hold the Monte Carlo histogram entries
        mc_weights = [] # define list to hold the Monte Carlo weights
        mc_colors = [] # define list to hold the colors of the Monte Carlo bars
        mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

        for s in samples: # loop over samples
            if s not in ['data']: # if not data
                mc_x.append( ak.to_numpy(data[s]['mll']) ) # append to the list of Monte Carlo histogram entries
                mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
            
        
        # *************
        # Main plot 
        # *************
        main_axes = plt.gca() # get current axes
        
        # plot the data points
        main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                        fmt='ko', # 'k' means black and 'o' is for circles 
                        label='Data') 
        
        # plot the Monte Carlo bars
        mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                    weights=mc_weights, stacked=True, 
                                    color=mc_colors, label=mc_labels )
        
        mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
        
        # calculate MC statistical uncertainty: sqrt(sum w^2)
        mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
        

        
        # plot the statistical uncertainty
        main_axes.bar(bin_centres, # x
                    2*mc_x_err, # heights
                    alpha=0.5, # half transparency
                    bottom=mc_x_tot-mc_x_err, color='none', 
                    hatch="////", width=step_size, label='Stat. Unc.' )

        # set the x-limit of the main axes
        main_axes.set_xlim( left=xmin, right=xmax ) 
        
        # separation of x axis minor ticks
        main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
        
        # set the axis tick parameters for the main axes
        main_axes.tick_params(which='both', # ticks on both x and y axes
                            direction='in', # Put ticks inside and outside the axes
                            top=True, # draw ticks on the top axis
                            right=True ) # draw ticks on right axis
        
        # x-axis label
        #main_axes.set_xlabel(r'2-lepton invariant mass $\mathrm{m_{2l}}$ [GeV]',
                            #fontsize=13, x=1, horizontalalignment='right' )
        
        # write y-axis label for main axes
        main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                            y=1, horizontalalignment='right') 
        
        # set y-axis limits for main axes
        main_axes.set_ylim( bottom=0.001, top=np.amax(data_x)*1.6 )
        
        # add minor ticks on y-axis for main axes
        main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05, # x
                0.93, # y
                'ATLAS Open Data', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                fontsize=13 ) 
        
        # Add text 'for education' on plot
        plt.text(0.05, # x
                0.88, # y
                'for education', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                style='italic',
                fontsize=8 ) 
        
        # Add energy and luminosity
        lumi_used = str(lumi*fraction) # luminosity to write on the plot
        plt.text(0.05, # x
                0.82, # y
                '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes
        
        # Add a label for the analysis carried out
        plt.text(0.05, # x
                0.76, # y
                r'$Z \rightarrow 2\ell$', # text 
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes

        # draw the legend
        main_axes.legend( frameon=False ) # no box around the legend
        
        #logarithmic y-axis
        main_axes.set_yscale('log')



        # *************
        # Data/MC ratio 
        # *************
        plt.axes([0.125,0.08,0.78,0.2]) #(left, bottom, width, height)
        ratio_axes = plt.gca()
        ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
        ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
            hatch="////",width=step_size)
        ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
        ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
        ratio_axes.set_xlabel(r'2-lepton invariant mass $\mathrm{m_{ll}}$ [GeV]',fontname='sans-serif',fontsize=11)
        ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
        ratio_axes.set_ylim(bottom=0,top=2.5)
        ratio_axes.set_yticks([0,1,2])
        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
        ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 
        
        return

    plot_data(data)
    #plt.savefig("mll.png")
    #plt.savefig("mll.pdf")
    plt.show()

    #Plotting Zpt
    def plot_data(data):

        xmin = 0 * GeV
        xmax = 200 * GeV
        step_size = 1 * GeV

        bin_edges = np.arange(start=xmin, # The interval includes this value
                        stop=xmax+step_size, # The interval doesn't include this value
                        step=step_size ) # Spacing between values
        bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                stop=xmax+step_size/2, # The interval doesn't include this value
                                step=step_size ) # Spacing between values

        data_x,_ = np.histogram(ak.to_numpy(data['data']['Zpt']), 
                                bins=bin_edges ) # histogram the data
        data_x_errors = np.sqrt( data_x ) # statistical error on the data


        mc_x = [] # define list to hold the Monte Carlo histogram entries
        mc_weights = [] # define list to hold the Monte Carlo weights
        mc_colors = [] # define list to hold the colors of the Monte Carlo bars
        mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

        for s in samples: # loop over samples
            if s not in ['data']: # if not data
                mc_x.append( ak.to_numpy(data[s]['Zpt']) ) # append to the list of Monte Carlo histogram entries
                mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
            
        
        # *************
        # Main plot 
        # *************
        main_axes = plt.gca() # get current axes
        
        # plot the data points
        main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                        fmt='ko', # 'k' means black and 'o' is for circles 
                        label='Data') 
        
        # plot the Monte Carlo bars
        mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                    weights=mc_weights, stacked=True, 
                                    color=mc_colors, label=mc_labels )
        
        mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
        
        # calculate MC statistical uncertainty: sqrt(sum w^2)
        mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
        

        
        # plot the statistical uncertainty
        main_axes.bar(bin_centres, # x
                    2*mc_x_err, # heights
                    alpha=0.5, # half transparency
                    bottom=mc_x_tot-mc_x_err, color='none', 
                    hatch="////", width=step_size, label='Stat. Unc.' )

        # set the x-limit of the main axes
        main_axes.set_xlim( left=xmin, right=xmax ) 
        
        # separation of x axis minor ticks
        main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
        
        # set the axis tick parameters for the main axes
        main_axes.tick_params(which='both', # ticks on both x and y axes
                            direction='in', # Put ticks inside and outside the axes
                            top=True, # draw ticks on the top axis
                            right=True ) # draw ticks on right axis
        
        # x-axis label
        #main_axes.set_xlabel(r'Zpt [GeV]',
                            #fontsize=13, x=1, horizontalalignment='right' )
        
        # write y-axis label for main axes
        main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                            y=1, horizontalalignment='right') 
        
        # set y-axis limits for main axes
        main_axes.set_ylim( bottom=10, top=np.amax(data_x)*1.6 )
        
        # add minor ticks on y-axis for main axes
        main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05, # x
                0.93, # y
                'ATLAS Open Data', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                fontsize=13 ) 
        
        # Add text 'for education' on plot
        plt.text(0.05, # x
                0.88, # y
                'for education', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                style='italic',
                fontsize=8 ) 
        
        # Add energy and luminosity
        lumi_used = str(lumi*fraction) # luminosity to write on the plot
        plt.text(0.05, # x
                0.82, # y
                '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes
        
        # Add a label for the analysis carried out
        plt.text(0.05, # x
                0.76, # y
                r'$Z \rightarrow 2\ell$', # text 
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes

        # draw the legend
        main_axes.legend( frameon=False ) # no box around the legend
        
        #logarithmic y-axis
        main_axes.set_yscale('log')

        # *************
        # Data/MC ratio 
        # *************
        plt.axes([0.122,0.08,0.83,0.2]) #(left, bottom, width, height)
        ratio_axes = plt.gca()
        ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
        ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
            hatch="////",width=step_size)
        ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
        ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
        ratio_axes.set_xlabel(r'Zpt [GeV]',fontname='sans-serif',fontsize=11)
        ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
        ratio_axes.set_ylim(bottom=0,top=2.5)
        ratio_axes.set_yticks([0,1,2])
        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
        ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 

        return

    plot_data(data)
    #plt.savefig("Zpt.png")
    #plt.savefig("Zpt.pdf")
    plt.show()

#Plotting MET
    def plot_data(data):

        xmin = 0 * GeV
        xmax = 200 * GeV
        step_size = 10 * GeV

        bin_edges = np.arange(start=xmin, # The interval includes this value
                        stop=xmax+step_size, # The interval doesn't include this value
                        step=step_size ) # Spacing between values
        bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                stop=xmax+step_size/2, # The interval doesn't include this value
                                step=step_size ) # Spacing between values

        data_x,_ = np.histogram(ak.to_numpy(data['data']['metpt']), 
                                bins=bin_edges ) # histogram the data
        data_x_errors = np.sqrt( data_x ) # statistical error on the data


        mc_x = [] # define list to hold the Monte Carlo histogram entries
        mc_weights = [] # define list to hold the Monte Carlo weights
        mc_colors = [] # define list to hold the colors of the Monte Carlo bars
        mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

        for s in samples: # loop over samples
            if s not in ['data']: # if not data
                mc_x.append( ak.to_numpy(data[s]['metpt']) ) # append to the list of Monte Carlo histogram entries
                mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
            
        
        # *************
        # Main plot 
        # *************
        main_axes = plt.gca() # get current axes
        
        # plot the data points
        main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                        fmt='ko', # 'k' means black and 'o' is for circles 
                        label='Data') 
        
        # plot the Monte Carlo bars
        mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                    weights=mc_weights, stacked=True, 
                                    color=mc_colors, label=mc_labels )
        
        mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
        
        # calculate MC statistical uncertainty: sqrt(sum w^2)
        mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
        

        
        # plot the statistical uncertainty
        main_axes.bar(bin_centres, # x
                    2*mc_x_err, # heights
                    alpha=0.5, # half transparency
                    bottom=mc_x_tot-mc_x_err, color='none', 
                    hatch="////", width=step_size, label='Stat. Unc.' )

        # set the x-limit of the main axes
        main_axes.set_xlim( left=xmin, right=xmax ) 
        
        # separation of x axis minor ticks
        main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
        
        # set the axis tick parameters for the main axes
        main_axes.tick_params(which='both', # ticks on both x and y axes
                            direction='in', # Put ticks inside and outside the axes
                            top=True, # draw ticks on the top axis
                            right=True ) # draw ticks on right axis
        
        # x-axis label
        #main_axes.set_xlabel(r'2-lepton invariant mass $\mathrm{m_{2l}}$ [GeV]',
                            #fontsize=13, x=1, horizontalalignment='right' )
        
        # write y-axis label for main axes
        main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                            y=1, horizontalalignment='right') 
        
        # set y-axis limits for main axes
        main_axes.set_ylim( bottom=0.001, top=np.amax(data_x)*1.6 )
        
        # add minor ticks on y-axis for main axes
        main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05, # x
                0.93, # y
                'ATLAS Open Data', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                fontsize=13 ) 
        
        # Add text 'for education' on plot
        plt.text(0.05, # x
                0.88, # y
                'for education', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                style='italic',
                fontsize=8 ) 
        
        # Add energy and luminosity
        lumi_used = str(lumi*fraction) # luminosity to write on the plot
        plt.text(0.05, # x
                0.82, # y
                '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes
        
        # Add a label for the analysis carried out
        plt.text(0.05, # x
                0.76, # y
                r'$Z \rightarrow 2\ell$', # text 
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes

        # draw the legend
        main_axes.legend( frameon=False ) # no box around the legend
        
        #logarithmic y-axis
        main_axes.set_yscale('log')



        # *************
        # Data/MC ratio 
        # *************
        plt.axes([0.125,0.08,0.78,0.2]) #(left, bottom, width, height)
        ratio_axes = plt.gca()
        ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
        ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
            hatch="////",width=step_size)
        ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
        ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
        ratio_axes.set_xlabel(r'MET [GeV]',fontname='sans-serif',fontsize=11)
        ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
        ratio_axes.set_ylim(bottom=0,top=2.5)
        ratio_axes.set_yticks([0,1,2])
        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
        ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 
        
        return

    plot_data(data)
    #plt.savefig("MET.png")
    #plt.savefig("MET.pdf")
    plt.show()

    #Plotting number of jets    
    def plot_data(data):

        xmin = 0 * GeV
        xmax = 15 * GeV
        step_size = 1 * GeV

        bin_edges = np.arange(start=xmin, # The interval includes this value
                        stop=xmax+step_size, # The interval doesn't include this value
                        step=step_size ) # Spacing between values
        bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                stop=xmax+step_size/2, # The interval doesn't include this value
                                step=step_size ) # Spacing between values

        data_x,_ = np.histogram(ak.to_numpy(data['data']['jet_n']), 
                                bins=bin_edges ) # histogram the data
        data_x_errors = np.sqrt( data_x ) # statistical error on the data


        mc_x = [] # define list to hold the Monte Carlo histogram entries
        mc_weights = [] # define list to hold the Monte Carlo weights
        mc_colors = [] # define list to hold the colors of the Monte Carlo bars
        mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

        for s in samples: # loop over samples
            if s not in ['data']: # if not data
                mc_x.append( ak.to_numpy(data[s]['jet_n']) ) # append to the list of Monte Carlo histogram entries
                mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
            
        
        # *************
        # Main plot 
        # *************
        main_axes = plt.gca() # get current axes
        
        # plot the data points
        main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                        fmt='ko', # 'k' means black and 'o' is for circles 
                        label='Data') 
        
        # plot the Monte Carlo bars
        mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                    weights=mc_weights, stacked=True, 
                                    color=mc_colors, label=mc_labels )
        
        mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
        
        # calculate MC statistical uncertainty: sqrt(sum w^2)
        mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
        

        
        # plot the statistical uncertainty
        main_axes.bar(bin_centres, # x
                    2*mc_x_err, # heights
                    alpha=0.5, # half transparency
                    bottom=mc_x_tot-mc_x_err, color='none', 
                    hatch="////", width=step_size, label='Stat. Unc.' )

        # set the x-limit of the main axes
        main_axes.set_xlim( left=xmin, right=xmax ) 
        
        # separation of x axis minor ticks
        main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
        
        # set the axis tick parameters for the main axes
        main_axes.tick_params(which='both', # ticks on both x and y axes
                            direction='in', # Put ticks inside and outside the axes
                            top=True, # draw ticks on the top axis
                            right=True ) # draw ticks on right axis
        
        # x-axis label
        #main_axes.set_xlabel(r'Number of Jets',
                            #fontsize=13, x=1, horizontalalignment='right' )
        
        # write y-axis label for main axes
        main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                            y=1, horizontalalignment='right') 
        
        # set y-axis limits for main axes
        main_axes.set_ylim( bottom=10, top=np.amax(data_x)*1.6 )
        
        # add minor ticks on y-axis for main axes
        main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05, # x
                0.93, # y
                'ATLAS Open Data', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                fontsize=13 ) 
        
        # Add text 'for education' on plot
        plt.text(0.05, # x
                0.88, # y
                'for education', # text
                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                style='italic',
                fontsize=8 ) 
        
        # Add energy and luminosity
        lumi_used = str(lumi*fraction) # luminosity to write on the plot
        plt.text(0.05, # x
                0.82, # y
                '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes
        
        # Add a label for the analysis carried out
        plt.text(0.05, # x
                0.76, # y
                r'$Jets$', # text 
                transform=main_axes.transAxes ) # coordinate system used is that of main_axes

        # draw the legend
        main_axes.legend( frameon=False ) # no box around the legend
        
        #logarithmic y-axis
        main_axes.set_yscale('log')

        # *************
        # Data/MC ratio 
        # *************
        plt.axes([0.122,0.08,0.855,0.2]) #(left, bottom, width, height)
        ratio_axes = plt.gca()
        ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
        ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
            hatch="////",width=step_size)
        ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
        ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
        ratio_axes.set_xlabel(r'Number of Jets',fontname='sans-serif',fontsize=11)
        ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
        ratio_axes.set_ylim(bottom=0,top=2.5)
        ratio_axes.set_yticks([0,1,2])
        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
        ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 


        return

    plot_data(data)
    #plt.savefig("Number of Jets.png")
    #plt.savefig("Number of Jets.pdf")
    plt.show()


if (type_of_study==2):

        #Ploting JZB
        def plot_data(data):

                xmin = -200 * GeV
                xmax = 200 * GeV
                step_size = 10 * GeV

                bin_edges = np.arange(start=xmin, # The interval includes this value
                                stop=xmax+step_size, # The interval doesn't include this value
                                step=step_size ) # Spacing between values
                bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                        stop=xmax+step_size/2, # The interval doesn't include this value
                                        step=step_size ) # Spacing between values

                data_x,_ = np.histogram(ak.to_numpy(data['data']['JZB']), 
                                        bins=bin_edges ) # histogram the data
                data_x_errors = np.sqrt( data_x ) # statistical error on the data


                mc_x = [] # define list to hold the Monte Carlo histogram entries
                mc_weights = [] # define list to hold the Monte Carlo weights
                mc_colors = [] # define list to hold the colors of the Monte Carlo bars
                mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

                for s in samples: # loop over samples
                        if s not in ['data']: # if not data
                                mc_x.append( ak.to_numpy(data[s]['JZB']) ) # append to the list of Monte Carlo histogram entries
                                mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
                
                
                # *************
                # Main plot 
                # *************
                main_axes = plt.gca() # get current axes
                
                # plot the data points
                main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                fmt='ko', # 'k' means black and 'o' is for circles 
                                label='Data') 
                
                # plot the Monte Carlo bars
                mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                        weights=mc_weights, stacked=True, 
                                        color=mc_colors, label=mc_labels )
                
                mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
                
                # calculate MC statistical uncertainty: sqrt(sum w^2)
                mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
                

                
                # plot the statistical uncertainty
                main_axes.bar(bin_centres, # x
                        2*mc_x_err, # heights
                        alpha=0.5, # half transparency
                        bottom=mc_x_tot-mc_x_err, color='none', 
                        hatch="////", width=step_size, label='Stat. Unc.' )

                # set the x-limit of the main axes
                main_axes.set_xlim( left=xmin, right=xmax ) 
                
                # separation of x axis minor ticks
                main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
                
                # set the axis tick parameters for the main axes
                main_axes.tick_params(which='both', # ticks on both x and y axes
                                direction='in', # Put ticks inside and outside the axes
                                top=True, # draw ticks on the top axis
                                right=True ) # draw ticks on right axis
                
                # x-axis label
                #main_axes.set_xlabel(r'2-lepton invariant mass $\mathrm{m_{2l}}$ [GeV]',
                                #fontsize=13, x=1, horizontalalignment='right' )
                
                # write y-axis label for main axes
                main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                                y=1, horizontalalignment='right') 
                
                # set y-axis limits for main axes
                main_axes.set_ylim( bottom=0.001, top=np.amax(data_x)*1.6 )
                
                # add minor ticks on y-axis for main axes
                main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

                # Add text 'ATLAS Open Data' on plot
                plt.text(0.05, # x
                        0.93, # y
                        'ATLAS Open Data', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        fontsize=13 ) 
                
                # Add text 'for education' on plot
                plt.text(0.05, # x
                        0.88, # y
                        'for education', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        style='italic',
                        fontsize=8 ) 
                
                # Add energy and luminosity
                lumi_used = str(lumi*fraction) # luminosity to write on the plot
                plt.text(0.05, # x
                        0.82, # y
                        '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes
                
                # Add a label for the analysis carried out
                plt.text(0.05, # x
                        0.76, # y
                        r'$Z \rightarrow 2\ell$', # text 
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes

                # draw the legend
                main_axes.legend( frameon=False ) # no box around the legend
                
                #logarithmic y-axis
                main_axes.set_yscale('log')



                # *************
                # Data/MC ratio 
                # *************
                plt.axes([0.125,0.08,0.78,0.2]) #(left, bottom, width, height)
                ratio_axes = plt.gca()
                ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
                ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
                hatch="////",width=step_size)
                ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
                ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
                ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
                ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
                ratio_axes.set_xlabel(r'JZB [GeV]',fontname='sans-serif',fontsize=11)
                ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
                ratio_axes.set_ylim(bottom=0,top=2.5)
                ratio_axes.set_yticks([0,1,2])
                ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
                ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 
                
                return

        plot_data(data)
        #plt.savefig("JZB.png")
        #plt.savefig("JZB.pdf")
        plt.show()

        #Plotting logos
        def plot_data(data):

                xmin = 0 * GeV
                xmax = 120 * GeV
                step_size = 1 * GeV

                bin_edges = np.arange(start=xmin, # The interval includes this value
                                stop=xmax+step_size, # The interval doesn't include this value
                                step=step_size ) # Spacing between values
                bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                        stop=xmax+step_size/2, # The interval doesn't include this value
                                        step=step_size ) # Spacing between values

                data_x,_ = np.histogram(ak.to_numpy(data['data']['logos']), 
                                        bins=bin_edges ) # histogram the data
                data_x_errors = np.sqrt( data_x ) # statistical error on the data


                mc_x = [] # define list to hold the Monte Carlo histogram entries
                mc_weights = [] # define list to hold the Monte Carlo weights
                mc_colors = [] # define list to hold the colors of the Monte Carlo bars
                mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

                for s in samples: # loop over samples
                        if s not in ['data']: # if not data
                                mc_x.append( ak.to_numpy(data[s]['logos']) ) # append to the list of Monte Carlo histogram entries
                                mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
                
                
                # *************
                # Main plot 
                # *************
                main_axes = plt.gca() # get current axes
                
                # plot the data points
                main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                fmt='ko', # 'k' means black and 'o' is for circles 
                                label='Data') 
                
                # plot the Monte Carlo bars
                mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                        weights=mc_weights, stacked=True, 
                                        color=mc_colors, label=mc_labels )
                
                mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
                
                # calculate MC statistical uncertainty: sqrt(sum w^2)
                mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
                

                
                # plot the statistical uncertainty
                main_axes.bar(bin_centres, # x
                        2*mc_x_err, # heights
                        alpha=0.5, # half transparency
                        bottom=mc_x_tot-mc_x_err, color='none', 
                        hatch="////", width=step_size, label='Stat. Unc.' )

                # set the x-limit of the main axes
                main_axes.set_xlim( left=xmin, right=xmax ) 
                
                # separation of x axis minor ticks
                main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
                
                # set the axis tick parameters for the main axes
                main_axes.tick_params(which='both', # ticks on both x and y axes
                                direction='in', # Put ticks inside and outside the axes
                                top=True, # draw ticks on the top axis
                                right=True ) # draw ticks on right axis
                
                # x-axis label
                #main_axes.set_xlabel(r'2-lepton invariant mass $\mathrm{m_{2l}}$ [GeV]',
                                #fontsize=13, x=1, horizontalalignment='right' )
                
                # write y-axis label for main axes
                main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                                y=1, horizontalalignment='right') 
                
                # set y-axis limits for main axes
                main_axes.set_ylim( bottom=0.001, top=np.amax(data_x)*1.6 )
                
                # add minor ticks on y-axis for main axes
                main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

                # Add text 'ATLAS Open Data' on plot
                plt.text(0.05, # x
                        0.93, # y
                        'ATLAS Open Data', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        fontsize=13 ) 
                
                # Add text 'for education' on plot
                plt.text(0.05, # x
                        0.88, # y
                        'for education', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        style='italic',
                        fontsize=8 ) 
                
                # Add energy and luminosity
                lumi_used = str(lumi*fraction) # luminosity to write on the plot
                plt.text(0.05, # x
                        0.82, # y
                        '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes
                
                # Add a label for the analysis carried out
                plt.text(0.05, # x
                        0.76, # y
                        r'$Z \rightarrow 2\ell$', # text 
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes

                # draw the legend
                main_axes.legend( frameon=False ) # no box around the legend
                
                #logarithmic y-axis
                main_axes.set_yscale('log')



                # *************
                # Data/MC ratio 
                # *************
                #plt.axes([0.125,0.08,0.78,0.2]) #(left, bottom, width, height)
                #ratio_axes = plt.gca()
                #ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
                #ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
                #hatch="////",width=step_size)
                #ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
                #ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
                #ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
                #ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
                #ratio_axes.set_xlabel(r'jet_pt/Zpt',fontname='sans-serif',fontsize=11)
                #ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
                #ratio_axes.set_ylim(bottom=0,top=2.5)
                #ratio_axes.set_yticks([0,1,2])
                #ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
                #ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 
                
                return

        plot_data(data)
        #plt.savefig("jetpt/Zpt.png")
        #plt.savefig("figure logos.pdf")
        plt.show()
        
        def plot_data(data):

                xmin = 0 * GeV
                xmax = 40 * GeV
                step_size = 1 * GeV

                bin_edges = np.arange(start=xmin, # The interval includes this value
                                stop=xmax+step_size, # The interval doesn't include this value
                                step=step_size ) # Spacing between values
                bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                        stop=xmax+step_size/2, # The interval doesn't include this value
                                        step=step_size ) # Spacing between values

                data_x,_ = np.histogram(ak.to_numpy(data['data']['logos2']), 
                                        bins=bin_edges ) # histogram the data
                data_x_errors = np.sqrt( data_x ) # statistical error on the data


                mc_x = [] # define list to hold the Monte Carlo histogram entries
                mc_weights = [] # define list to hold the Monte Carlo weights
                mc_colors = [] # define list to hold the colors of the Monte Carlo bars
                mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

                for s in samples: # loop over samples
                        if s not in ['data']: # if not data
                                mc_x.append( ak.to_numpy(data[s]['logos2']) ) # append to the list of Monte Carlo histogram entries
                                mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
                
                
                # *************
                # Main plot 
                # *************
                main_axes = plt.gca() # get current axes
                
                # plot the data points
                main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                fmt='ko', # 'k' means black and 'o' is for circles 
                                label='Data') 
                
                # plot the Monte Carlo bars
                mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                        weights=mc_weights, stacked=True, 
                                        color=mc_colors, label=mc_labels )
                
                mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
                
                # calculate MC statistical uncertainty: sqrt(sum w^2)
                mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
                

                
                # plot the statistical uncertainty
                main_axes.bar(bin_centres, # x
                        2*mc_x_err, # heights
                        alpha=0.5, # half transparency
                        bottom=mc_x_tot-mc_x_err, color='none', 
                        hatch="////", width=step_size, label='Stat. Unc.' )

                # set the x-limit of the main axes
                main_axes.set_xlim( left=xmin, right=xmax ) 
                
                # separation of x axis minor ticks
                main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
                
                # set the axis tick parameters for the main axes
                main_axes.tick_params(which='both', # ticks on both x and y axes
                                direction='in', # Put ticks inside and outside the axes
                                top=True, # draw ticks on the top axis
                                right=True ) # draw ticks on right axis
                
                # x-axis label
                #main_axes.set_xlabel(r'2-lepton invariant mass $\mathrm{m_{2l}}$ [GeV]',
                                #fontsize=13, x=1, horizontalalignment='right' )
                
                # write y-axis label for main axes
                main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                                y=1, horizontalalignment='right') 
                
                # set y-axis limits for main axes
                main_axes.set_ylim( bottom=0.001, top=np.amax(data_x)*1.6 )
                
                # add minor ticks on y-axis for main axes
                main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

                # Add text 'ATLAS Open Data' on plot
                plt.text(0.05, # x
                        0.93, # y
                        'ATLAS Open Data', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        fontsize=13 ) 
                
                # Add text 'for education' on plot
                plt.text(0.05, # x
                        0.88, # y
                        'for education', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        style='italic',
                        fontsize=8 ) 
                
                # Add energy and luminosity
                lumi_used = str(lumi*fraction) # luminosity to write on the plot
                plt.text(0.05, # x
                        0.82, # y
                        '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes
                
                # Add a label for the analysis carried out
                plt.text(0.05, # x
                        0.76, # y
                        r'$Z \rightarrow 2\ell$', # text 
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes

                # draw the legend
                main_axes.legend( frameon=False ) # no box around the legend
                
                #logarithmic y-axis
                main_axes.set_yscale('log')



                # *************
                # Data/MC ratio 
                # *************
                #plt.axes([0.125,0.08,0.78,0.2]) #(left, bottom, width, height)
                #ratio_axes = plt.gca()
                #ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
                #ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
                #hatch="////",width=step_size)
                #ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
                #ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
                #ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
                #ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
                #ratio_axes.set_xlabel(r'jet_pt/Zpt',fontname='sans-serif',fontsize=11)
                #ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
                #ratio_axes.set_ylim(bottom=0,top=2.5)
                #ratio_axes.set_yticks([0,1,2])
                #ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
                #ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 
                
                return

        plot_data(data)
        #plt.savefig("jetpt/Zpt.png")
        #plt.savefig("figure logos2.pdf")
        plt.show()

        #Plotting leading jet pt
    
        def plot_data(data):

                xmin = 20 * GeV
                xmax = 80 * GeV
                step_size = 1 * GeV

                bin_edges = np.arange(start=xmin, # The interval includes this value
                                stop=xmax+step_size, # The interval doesn't include this value
                                step=step_size ) # Spacing between values
                bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                        stop=xmax+step_size/2, # The interval doesn't include this value
                                        step=step_size ) # Spacing between values

                data_x,_ = np.histogram(ak.to_numpy(data['data']['jetpt_leading']), 
                                        bins=bin_edges ) # histogram the data
                data_x_errors = np.sqrt( data_x ) # statistical error on the data


                mc_x = [] # define list to hold the Monte Carlo histogram entries
                mc_weights = [] # define list to hold the Monte Carlo weights
                mc_colors = [] # define list to hold the colors of the Monte Carlo bars
                mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

                for s in samples: # loop over samples
                        if s not in ['data']: # if not data
                                mc_x.append(ak.to_numpy(data[s]['jetpt_leading'])) # append to the list of Monte Carlo histogram entries
                                mc_weights.append(ak.to_numpy(data[s].totalWeight)) # append to the list of Monte Carlo weights
                                mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                                mc_labels.append( s ) # append to the list of Monte Carlo legend labels  

                #np.resize(ak.to_numpy(ak.pad_none(data[s]['jetpt'], 1, axis=1, clip=True)),(1,1))    
                #np.resize((ak.to_numpy(data[s].totalWeight )),(1,1)) /ak.to_numpy(ak.pad_none(data[s].totalWeight, 1, axis=0, clip=True)).flatten()
                # *************
                # Main plot 
                # *************
                main_axes = plt.gca() # get current axes
                
                # plot the data points
                main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                fmt='ko', # 'k' means black and 'o' is for circles 
                                label='Data') 
                
                # plot the Monte Carlo bars
                mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                        weights=mc_weights, stacked=True, 
                                        color=mc_colors, label=mc_labels )
                
                mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
                
                # calculate MC statistical uncertainty: sqrt(sum w^2)
                mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
                

                
                # plot the statistical uncertainty
                main_axes.bar(bin_centres, # x
                        2*mc_x_err, # heights
                        alpha=0.5, # half transparency
                        #bottom=mc_x_tot-mc_x_err,
                        color='none', 
                        hatch="////", width=step_size, label='Stat. Unc.' )

                # set the x-limit of the main axes
                main_axes.set_xlim( left=xmin, right=xmax ) 
                
                # separation of x axis minor ticks
                main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
                
                # set the axis tick parameters for the main axes
                main_axes.tick_params(which='both', # ticks on both x and y axes
                                direction='in', # Put ticks inside and outside the axes
                                top=True, # draw ticks on the top axis
                                right=True ) # draw ticks on right axis
                
                # x-axis label
                #main_axes.set_xlabel(r'jet_pt [GeV]',
                #                   fontsize=13, x=1, horizontalalignment='right' )
                
                # write y-axis label for main axes
                main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                                y=1, horizontalalignment='right') 
                
                # set y-axis limits for main axes
                main_axes.set_ylim( bottom=1, top=np.amax(data_x)*1.6 )
                
                # add minor ticks on y-axis for main axes
                main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

                # Add text 'ATLAS Open Data' on plot
                plt.text(0.05, # x
                        0.93, # y
                        'ATLAS Open Data', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        fontsize=13 ) 
                
                # Add text 'for education' on plot
                plt.text(0.05, # x
                        0.88, # y
                        'for education', # text
                        transform=main_axes.transAxes, # coordinate system used is that of main_axes
                        style='italic',
                        fontsize=8 ) 
                
                # Add energy and luminosity
                lumi_used = str(lumi*fraction) # luminosity to write on the plot
                plt.text(0.05, # x
                        0.82, # y
                        '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes
                
                # Add a label for the analysis carried out
                plt.text(0.05, # x
                        0.76, # y
                        r'$Z \rightarrow 2\ell$', # text 
                        transform=main_axes.transAxes ) # coordinate system used is that of main_axes

                # draw the legend
                main_axes.legend( frameon=False ) # no box around the legend
                
                #logarithmic y-axis
                main_axes.set_yscale('log')

                # *************
                # Data/MC ratio 
                # *************
                plt.axes([0.125,0.08,0.78,0.2]) #(left, bottom, width, height)
                ratio_axes = plt.gca()
                ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
                ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
                hatch="////",width=step_size)
                ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
                ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
                ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
                ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
                ratio_axes.set_xlabel(r'jet_pt [GeV]',fontname='sans-serif',fontsize=11)
                ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
                ratio_axes.set_ylim(bottom=0,top=2.5)
                ratio_axes.set_yticks([0,1,2])
                ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
                ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 

                return

        plot_data(data)
        #plt.savefig("jetpt_leading.png")
        #plt.savefig("jetpt_leading.pdf")
        plt.show()

        if (num==2):
                #Plotting Mjj
                def plot_data(data):

                        xmin = 10 * GeV
                        xmax = 200 * GeV
                        step_size = 1 * GeV

                        bin_edges = np.arange(start=xmin, # The interval includes this value
                                        stop=xmax+step_size, # The interval doesn't include this value
                                        step=step_size ) # Spacing between values
                        bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                                                stop=xmax+step_size/2, # The interval doesn't include this value
                                                step=step_size ) # Spacing between values

                        data_x,_ = np.histogram(ak.to_numpy(data['data']['mjj']), 
                                                bins=bin_edges ) # histogram the data
                        data_x_errors = np.sqrt( data_x ) # statistical error on the data


                        mc_x = [] # define list to hold the Monte Carlo histogram entries
                        mc_weights = [] # define list to hold the Monte Carlo weights
                        mc_colors = [] # define list to hold the colors of the Monte Carlo bars
                        mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

                        for s in samples: # loop over samples
                                if s not in ['data']: # if not data
                                        mc_x.append( ak.to_numpy(data[s]['mjj']) ) # append to the list of Monte Carlo histogram entries
                                        mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
                                        mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
                                        mc_labels.append( s ) # append to the list of Monte Carlo legend labels  
                        
                        
                        # *************
                        # Main plot 
                        # *************
                        main_axes = plt.gca() # get current axes
                        
                        # plot the data points
                        main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                        fmt='ko', # 'k' means black and 'o' is for circles 
                                        label='Data') 
                        
                        # plot the Monte Carlo bars
                        mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                                weights=mc_weights, stacked=True, 
                                                color=mc_colors, label=mc_labels )
                        
                        mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
                        
                        # calculate MC statistical uncertainty: sqrt(sum w^2)
                        mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
                        

                        
                        # plot the statistical uncertainty
                        main_axes.bar(bin_centres, # x
                                2*mc_x_err, # heights
                                alpha=0.5, # half transparency
                                bottom=mc_x_tot-mc_x_err, color='none', 
                                hatch="////", width=step_size, label='Stat. Unc.' )

                        # set the x-limit of the main axes
                        main_axes.set_xlim( left=xmin, right=xmax ) 
                        
                        # separation of x axis minor ticks
                        main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
                        
                        # set the axis tick parameters for the main axes
                        main_axes.tick_params(which='both', # ticks on both x and y axes
                                        direction='in', # Put ticks inside and outside the axes
                                        top=True, # draw ticks on the top axis
                                        right=True ) # draw ticks on right axis
                        
                        # x-axis label
                        #main_axes.set_xlabel(r'2-lepton invariant mass $\mathrm{m_{2l}}$ [GeV]',
                                        #fontsize=13, x=1, horizontalalignment='right' )
                        
                        # write y-axis label for main axes
                        main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                                        y=1, horizontalalignment='right') 
                        
                        # set y-axis limits for main axes
                        main_axes.set_ylim( bottom=0.001, top=np.amax(data_x)*1.6 )
                        
                        # add minor ticks on y-axis for main axes
                        main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

                        # Add text 'ATLAS Open Data' on plot
                        plt.text(0.05, # x
                                0.93, # y
                                'ATLAS Open Data', # text
                                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                                fontsize=13 ) 
                        
                        # Add text 'for education' on plot
                        plt.text(0.05, # x
                                0.88, # y
                                'for education', # text
                                transform=main_axes.transAxes, # coordinate system used is that of main_axes
                                style='italic',
                                fontsize=8 ) 
                        
                        # Add energy and luminosity
                        lumi_used = str(lumi*fraction) # luminosity to write on the plot
                        plt.text(0.05, # x
                                0.82, # y
                                '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
                                transform=main_axes.transAxes ) # coordinate system used is that of main_axes
                        
                        # Add a label for the analysis carried out
                        plt.text(0.05, # x
                                0.76, # y
                                r'$Z \rightarrow 2\ell$', # text 
                                transform=main_axes.transAxes ) # coordinate system used is that of main_axes

                        # draw the legend
                        main_axes.legend( frameon=False ) # no box around the legend
                        
                        #logarithmic y-axis
                        main_axes.set_yscale('log')



                        # *************
                        # Data/MC ratio 
                        # *************
                        plt.axes([0.125,0.08,0.78,0.2]) #(left, bottom, width, height)
                        ratio_axes = plt.gca()
                        ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
                        ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
                        hatch="////",width=step_size)
                        ratio_axes.plot(bin_edges,np.ones(len(bin_edges)),color='k')
                        ratio_axes.set_xlim(left=xmin,right=bin_edges[-1])
                        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
                        ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
                        ratio_axes.set_xlabel(r'2-jet invariant mass $\mathrm{m_{jj}}$ [GeV]',fontname='sans-serif',fontsize=11)
                        ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
                        ratio_axes.set_ylim(bottom=0,top=2.5)
                        ratio_axes.set_yticks([0,1,2])
                        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
                        ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11) 
                        
                        return

                plot_data(data)
                #plt.savefig("mjj.png")
                #plt.savefig("mjj.pdf")
                plt.show()
