import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
import astropy.wcs
import scipy.spatial

#doPCAT = False

psffull = np.loadtxt('Data/sdss.0921_psf.txt', skiprows=1)
psf0 = psffull[0:125:5, 0:125:5]
def err_f(f):
        back = 179
        gain = 4.62
        return 1./np.sqrt(gain*np.sum(psf0*psf0/(back+psf0*f)))
def err_mag(mag):
        f = 10**((22.5 - mag)/2.5) / 0.00546689
        return 1.08573620476 * np.sqrt((err_f(f) / f)**2 + 0.01**2)
def adutomag(adu):
        return 22.5 - 2.5 * np.log10(0.00546689 * adu)

hwhm=2.5
'''DPcat = np.loadtxt('Data/m2_2583.phot')
DPx = DPcat[:,20]
DPy = DPcat[:,21]
DPr = DPcat[:,22]
DPrerr = DPcat[:,23]
DPchi = DPcat[:,24]
DPsharp = DPcat[:,25]
DPx -= 310 + 1
DPy -= 630 + 1362
mask = np.logical_and(np.logical_and(DPx > 0+hwhm, DPx < 99-hwhm), np.logical_and(DPy > 0+hwhm, DPy < 99-hwhm))
DPx = DPx[mask]
DPy = DPy[mask]
DPr = DPr[mask]
DPrerr = DPrerr[mask]
DPchi = DPchi[mask]
DPsharp = DPsharp[mask]
DPexprerr = np.array([err_mag(i) for i in DPr])
DPs = DPrerr / DPexprerr
DPc = np.zeros((DPx.shape[0], 2))
DPc[:,0] = DPx
DPc[:,1] = DPy
DPkd = scipy.spatial.KDTree(DPc)'''

HTcat = np.loadtxt('../pcat-dnest/Data/NGC7089R.RDVIQ.cal.adj.zpt', skiprows=1)
HTra = HTcat[:,21]
HTdc = HTcat[:,22]
HT606= HTcat[:,3]
HT606err = HTcat[:,4]
HT814= HTcat[:,7]
hdulist = astropy.io.fits.open('../pcat-dnest/Data/frame-r-002583-2-0136.fits')
w = astropy.wcs.WCS(hdulist['PRIMARY'].header)
pix_coordinates = w.wcs_world2pix(HTra, HTdc, 0)
HTx = pix_coordinates[0] - 310
HTy = pix_coordinates[1] - 630
mask = np.logical_and(HT606 > 0, HT814 > 0)
mask = np.logical_and(np.logical_and(np.logical_and(HTx > 0+hwhm, HTx < 99-hwhm), np.logical_and(HTy > 0+hwhm, HTy < 99-hwhm)), mask)
HTx = HTx[mask]
HTy = HTy[mask]
HT606 = HT606[mask]
HT606err = HT606err[mask]
HT814 = HT814[mask]
HTc = np.zeros((HTx.shape[0], 2))
HTc[:, 0] = HTx
HTc[:, 1] = HTy
HTkd = scipy.spatial.KDTree(HTc)

print np.sum(HT606 < 22), 'HST brighter than 21'

'''CCcat = np.loadtxt('run-alpha-20-new/condensed_catalog.txt')
CCx = CCcat[:,0]
CCy = CCcat[:,2]
CCr = adutomag(CCcat[:,4])
CCrerr = 1.086 * CCcat[:,5] / CCcat[:,4]
CCconf = CCcat[:,8]
CCs = CCcat[:,11]
mask = np.logical_and(np.logical_and(CCx > 0+hwhm, CCx < 99-hwhm), np.logical_and(CCy > 0+hwhm, CCy < 99-hwhm))
mask = np.logical_and(mask, CCcat[:,4] > 0)
CCx = CCx[mask]
CCy = CCy[mask]
CCr = CCr[mask]
CCs = CCs[mask]
CCconf = CCconf[mask]
CCc = np.zeros((CCx.shape[0], 2))
CCc[:,0] = CCx
CCc[:,1] = CCy
CCkd = scipy.spatial.KDTree(CCc)'''

PCcat = np.loadtxt('../pcat-dnest/run-alpha-20-new/posterior_sample.txt')
#PCcat = np.loadtxt('run-0923/posterior_sample.txt')
maxn = 3000
PCn = PCcat[:,10003].astype(np.int)
PCx = PCcat[:,10004:10004+maxn]
PCy = PCcat[:,10004+maxn:10004+2*maxn]
PCf = PCcat[:,10004+2*maxn:10004+3*maxn]
PCr = adutomag(PCf)
mask = (PCf > 0) * (PCx > 0+hwhm) * (PCx < 99-hwhm) * (PCy > 0+hwhm) * (PCy < 99-hwhm)
PCc_all = np.zeros((np.sum(mask), 2))
PCc_all[:, 0] = PCx[mask].flatten()
PCc_all[:, 1] = PCy[mask].flatten()
PCr_all = PCr[mask].flatten()
PCkd = scipy.spatial.KDTree(PCc_all)

print np.mean(PCn), 'mean PCAT sources'

lion = np.load('chain-sdss.0921L-long.npz')
PC2n = lion['n'][-300:].astype(np.int)
PC2x = lion['x'][-300:,:]
PC2y = lion['y'][-300:,:]
PC2f = lion['f'][-300:,:]
PC2r = adutomag(PC2f)
mask = (PC2f > 0) * (PC2x > 0+hwhm) * (PC2x < 99-hwhm) * (PC2y > 0+hwhm) * (PC2y < 99-hwhm)
PC2c_all = np.zeros((np.sum(mask), 2))
PC2c_all[:, 0] = PC2x[mask].flatten()
PC2c_all[:, 1] = PC2y[mask].flatten()
PC2r_all = PC2r[mask].flatten()
PC2kd = scipy.spatial.KDTree(PC2c_all)


sizefac = 1360.
n = PCn[-1]
plt.scatter(PCx[-1,0:n], PCy[-1,0:n], s=PCf[-1,0:n]/sizefac, c='r', marker='x')
n = PC2n[-1]
plt.scatter(PC2x[-1,0:n], PC2y[-1,0:n], s=PC2f[-1,0:n]/sizefac, c='m', marker='+')
plt.show()


def associate(a, mags_a, b, mags_b, dr, dmag, confs_b = None, sigfs_b = None):
	allmatches = a.query_ball_tree(b, dr)
	goodmatch = np.zeros(mags_a.size, np.bool)
	if confs_b is not None:
		confmatch = np.zeros(mags_a.size)
	if sigfs_b is not None:
		sigfmatch = np.zeros(mags_a.size) + float('inf')

	for i in xrange(len(allmatches)):
		matches = allmatches[i]
		if len(matches):
			mag_a = mags_a[i]
			goodmatch[i] = False
			for j in matches:
				mag_b = mags_b[j]
				if np.abs(mag_a - mag_b) < dmag:
					goodmatch[i] = True
					if (confs_b is not None) and (confs_b[j] > confmatch[i]):
						confmatch[i] = confs_b[j]
					if (sigfs_b is not None) and (sigfs_b[j] < sigfmatch[i]):
						sigfmatch[i] = sigfs_b[j]

	if confs_b is not None:
		if sigfs_b is not None:
			return goodmatch, confmatch, sigfmatch
		else:
			return goodmatch, confmatch
	else:
		if sigfs_b is not None:
			return goodmatch, sigfmatch
		else:
			return goodmatch

dr = 0.75
dmag = 0.5

nbins = 16
minr, maxr = 15.5, 23.5
binw = (maxr - minr) / float(nbins)
#print "max r DP", np.max(DPr)
#print "max r CC", np.max(CCr)

precPC = np.zeros(nbins)
precPC2= np.zeros(nbins)
#precCC = np.zeros(nbins)
#precDP = np.zeros(nbins)

# with sigfac < 8
#precCCs = np.zeros(nbins)
#precDPs = np.zeros(nbins)

goodmatchPC = associate(PCkd, PCr_all, HTkd, HT606, dr, dmag)
goodmatchPC2= associate(PC2kd, PC2r_all, HTkd, HT606, dr, dmag)
#goodmatchDP = associate(DPkd, DPr, HTkd, HT606, dr, dmag)
#goodmatchCC = associate(CCkd, CCr, HTkd, HT606, dr, dmag)


for i in xrange(nbins):
    rlo = minr + i * binw
    rhi = rlo + binw

	#inbin = np.logical_and(DPr >= rlo, DPr < rhi)
	#print i, np.sum(inbin)
	#precDP[i] = np.sum(np.logical_and(inbin, goodmatchDP)) / float(np.sum(inbin))
	#inbin = np.logical_and(inbin, DPs < 8)
	#precDPs[i] = np.sum(np.logical_and(inbin, goodmatchDP)) / float(np.sum(inbin))

    inbin = np.logical_and(PCr_all >= rlo, PCr_all < rhi)
    precPC[i] = np.sum(np.logical_and(inbin, goodmatchPC)) / float(np.sum(inbin))

    inbin = np.logical_and(PC2r_all >= rlo, PC2r_all < rhi)
    precPC2[i] = np.sum(np.logical_and(inbin, goodmatchPC2)) / float(np.sum(inbin))

	#inbin = np.logical_and(CCr >= rlo, CCr < rhi)
	#precCC[i] = np.sum(CCconf[np.logical_and(inbin, goodmatchCC)]) / float(np.sum(CCconf[inbin]))
	#inbin = np.logical_and(inbin, CCs < 8)
	#precCCs[i] = np.sum(CCconf[np.logical_and(inbin, goodmatchCC)]) / float(np.sum(CCconf[inbin]))

#plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precDP, c='b', label='DAOPHOT Catalog', marker='+', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC, c='r', label='Portillo et al. (2017)', marker='x', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC2, c='m', label='this work', marker='+', markersize=10, mew=2)
plt.xlabel('SDSS r magnitude')
plt.ylabel('false discovery rate')
plt.ylim((-0.05, 0.7))
plt.xlim((15,24))
plt.legend(prop={'size':12}, loc = 'best')
plt.savefig('fdr-lion.pdf')
plt.show()

'''	plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precDP, c='b', label='DAOPHOT Catalog', marker='+', markersize=10, mew=2)
	plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precPC, c='r', label='Catalog Ensemble', marker='x', markersize=10, mew=2)
	plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precCC, c='purple', label='Condensed Catalog', marker='1', markersize=10, mew=2)
	plt.xlabel('SDSS r magnitude')
	plt.ylabel('false discovery rate')
	plt.ylim((-0.05, 0.7))
	plt.xlim((15,24))
	plt.legend(prop={'size':12}, loc = 'best')
	plt.savefig('fdr_classical.pdf')
	plt.show()

plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precDP, c='b', label='DAOPHOT Catalog', marker='+', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precDPs, c='b', ls='--', label='with DF < 8', marker='+', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precCC, c='purple', label='Condensed Catalog', marker='1', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-precCCs, c='purple', ls='--', label='with DF < 8', marker='1', markersize=10, mew=2)
plt.xlabel('SDSS r magnitude')
plt.ylabel('false discovery rate')
plt.ylim((-0.05, 0.7))
plt.xlim((15,24))
plt.legend(prop={'size':12}, loc = 'best')
plt.savefig('fdr_sigfac.pdf')
plt.show()'''

nbins = 16
minr, maxr = 15.5, 23.5
binw = (maxr - minr) / float(nbins)

#completeDP, sigfDP = associate(HTkd, HT606, DPkd, DPr, dr, dmag, sigfs_b=DPs)
#completeCC, confmatchCC, sigfCC = associate(HTkd, HT606, CCkd, CCr, dr, dmag, confs_b=CCconf, sigfs_b=CCs)

#if doPCAT:
completePC = np.zeros((PCx.shape[0], HTx.size))
for i in xrange(PCx.shape[0]):
    print i
    n = PCn[i]
    CCc_one = np.zeros((n,2))
    CCc_one[:, 0] = PCx[i, 0:n]
    CCc_one[:, 1] = PCy[i, 0:n]
    CCr_one = PCr[i, 0:n]
    completePC[i, :] = associate(HTkd, HT606, scipy.spatial.KDTree(CCc_one), CCr_one, dr, dmag)
completePC = np.sum(completePC, axis=0) / float(PCx.shape[0])

completePC2 = np.zeros((PC2x.shape[0], HTx.size))
for i in xrange(PCx.shape[0]):
        print 'B', i
        n = PC2n[i]
        CCc_one = np.zeros((n,2))
        CCc_one[:, 0] = PC2x[i,0:n]
        CCc_one[:, 1] = PC2y[i,0:n]
        CCr_one = PC2r[i, 0:n]
        completePC2[i,:] = associate(HTkd, HT606, scipy.spatial.KDTree(CCc_one), CCr_one, dr, dmag)
completePC2 = np.sum(completePC2, axis=0) / float(PC2x.shape[0])

#reclDP = np.zeros(nbins)
reclPC = np.zeros(nbins)
reclPC2= np.zeros(nbins)
#reclCC = np.zeros(nbins)

#reclDPs = np.zeros(nbins)
#reclCCs = np.zeros(nbins)

for i in xrange(nbins):
        rlo = minr + i * binw
        rhi = rlo + binw

        inbin = np.logical_and(HT606 >= rlo, HT606 < rhi)

#	if doPCAT:
        reclPC[i] = np.sum(completePC[inbin]) / float(np.sum(inbin))
        reclPC2[i] = np.sum(completePC2[inbin]) / float(np.sum(inbin))

#        reclDP[i] = np.sum(np.logical_and(inbin, completeDP)) / float(np.sum(inbin))
#        completeDPs = np.logical_and(completeDP, sigfDP < 8)
#        reclDPs[i] = np.sum(np.logical_and(inbin, completeDPs)) / float(np.sum(inbin))

#	reclCC[i] = np.sum(confmatchCC[inbin]) / float(np.sum(inbin))
#	reclCCs[i] = np.sum(confmatchCC[np.logical_and(inbin, sigfCC < 8)]) / float(np.sum(inbin))

#plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclDP, 'b--', label='DAOPHOT Catalog', marker='+', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC, c='r', label='Portillo et al. (2017)', marker='x', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC2, c='m', label='this work', marker='+', markersize=10, mew=2)
plt.xlabel('HST F606W magnitude', fontsize='large')
plt.ylabel('completeness', fontsize='large')
plt.ylim((-0.1,1.1))
#plt.legend(prop={'size':12}, loc = 'best')
plt.legend(loc='best', fontsize='large')
plt.savefig('completeness-lion.pdf')
plt.show()

'''	plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclDP, c='b', label='DAOPHOT Catalog', marker='+', markersize=10, mew=2)
	plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC, c='r', label='Catalog Ensemble', marker='x', markersize=10, mew=2)
	plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclCC, c='purple', label='Condensed Catalog', marker='1', markersize=10, mew=2)
	plt.xlabel('HST F606W magnitude')
	plt.ylabel('completeness')
	plt.ylim((-0.1, 1.1))
	plt.legend(prop={'size':12}, loc = 'best')
	plt.savefig('completeness_classical.pdf')
	plt.show()

plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclDP, c='b', label='DAOPHOT Catalog', marker='+', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclDPs, c='b', ls='--', label='with DF < 8', marker='+', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclCC, c='purple', label='Condensed Catalog', marker='1', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclCCs, c='purple', ls='--', label='with DF < 8', marker='1', markersize=10, mew=2)
plt.ylim((-0.1, 1.1))
plt.xlabel('HST F606W magnitude')
plt.ylabel('completeness')
plt.legend(prop={'size':12}, loc = 'best')
plt.savefig('completeness_sigfac.pdf')
plt.show()

nbins = 4
nsigf = 71
minr, maxr = 17.5, 21.5
binw = (maxr - minr) / float(nbins)
print np.log10(max(np.max(DPs), np.max(CCs)))
sigfaccuts = np.logspace(0, 1.4, num=nsigf)

#table
reclCCt = np.zeros((nbins, nsigf))
precCCt = np.zeros((nbins, nsigf))
reclDPt = np.zeros((nbins, nsigf))
precDPt = np.zeros((nbins, nsigf))

for i in xrange(nbins):
	rlo = minr + i * binw
	rhi = rlo + binw

	inbinHT = np.logical_and(HT606 >= rlo, HT606 < rhi)
	inbinDP = np.logical_and(DPr >= rlo, DPr < rhi)
	inbinCC = np.logical_and(CCr >= rlo, CCr < rhi)
	print 'mag', (rlo+rhi)/2., 'n', np.sum(inbinHT)
	for j in xrange(nsigf):
		sigfaccut = sigfaccuts[j]

		reclDPt[i,j] = np.sum(np.logical_and(inbinHT, np.logical_and(completeDP, sigfDP < sigfaccut))) / float(np.sum(inbinHT))
		reclCCt[i,j] = np.sum(confmatchCC[np.logical_and(inbinHT, sigfCC < sigfaccut)]) / float(np.sum(inbinHT))

		precDPt[i,j] = np.sum(np.logical_and(inbinDP, np.logical_and(goodmatchDP, DPs < sigfaccut))) / float(np.sum(np.logical_and(inbinDP, DPs < sigfaccut)))
		precCCt[i,j] = np.sum(CCconf[np.logical_and(inbinCC, np.logical_and(goodmatchCC, CCs < sigfaccut))]) / float(np.sum(CCconf[np.logical_and(inbinCC, CCs < sigfaccut)]))


linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
markers = ['o', 'v', 's', 'D', 'p', '*']
#markers = ['$1.8$', '$3.2$', '$5.6$', '$10$', '$18$']

for i in xrange(nbins):
        rlo = minr + i * binw
	rme = rlo + 0.5*binw
        rhi = rlo + binw

	if i == 0:
		label1 = 'DAOPHOT Catalog mag %2.0f' % rme
		label2 = 'Condensed Catalog mag %2.0f' % rme
	else:
		label1 = None
		label2 = 'mag %2.0f' % rme
	# make it convex: more stringent cuts that worsen precision are not used
	for j in xrange(nsigf-2, -1, -1):
		if precDPt[i,j] < precDPt[i,j+1]:
			precDPt[i,j] = precDPt[i,j+1]
			reclDPt[i,j] = reclDPt[i,j+1]
		if precCCt[i,j] < precCCt[i,j+1]:
			precCCt[i,j] = precCCt[i,j+1]
			reclCCt[i,j] = reclCCt[i,j+1]
	
	# repeats make it maximally convex as allowed by data points
	plt.plot(1-np.repeat(precDPt[i,:], 2)[:-1], np.repeat(reclDPt[i,:], 2)[1:], c='b', ls=linestyles[i], label=label1, zorder=1)
	plt.plot(1-np.repeat(precCCt[i,:], 2)[:-1], np.repeat(reclCCt[i,:], 2)[1:], c='purple', ls=linestyles[i], label=label2, zorder=2)

for i in xrange(nbins):
	for j in xrange(len(markers)):
		if i == 0:
			if j == 0:
				label3 = 'DF = %2.1f' % sigfaccuts[10*j+10]
			else:
				label3 = '%2.1f' % sigfaccuts[10*j+10]
		else:
			label3 = None
		plt.scatter(1-precDPt[i,10*j+10], reclDPt[i,10*j+10], c='b', marker=markers[j], s=100, zorder=1)
		plt.scatter(1-precCCt[i,10*j+10], reclCCt[i,10*j+10], c='purple', marker=markers[j], s=100, label=label3, zorder=2)
plt.xlabel('false discovery rate')
plt.ylabel('completeness')
plt.xlim((-0.025,0.4))
plt.ylim((-0.1,1.1))
art = [plt.legend(prop={'size':12}, loc='upper center', bbox_to_anchor=(0.5, -0.1), scatterpoints=1, ncol=3)]
plt.savefig('roc.pdf', additional_artists=art, bbox_inches='tight')
plt.show()

image = np.loadtxt('Data/sdss.0921_cts.txt')
plt.imshow(image, cmap='Greys', origin='lower', interpolation='none', vmin=np.min(image), vmax=np.percentile(image, 97.5))
plt.colorbar()
plt.xlim((-0.5,99.5))
plt.ylim((-0.5,99.5))
plt.tight_layout()
plt.savefig('crowded_blank.png')
plt.show()
plt.imshow(image, cmap='Greys', origin='lower', interpolation='none', vmin=np.min(image), vmax=np.percentile(image, 97.5))
plt.colorbar()
plt.scatter(DPx, DPy, marker='o', s=400, facecolors='none', edgecolors='b')
plt.xlim((-0.5,99.5))
plt.ylim((-0.5,99.5))
plt.tight_layout()
plt.savefig('crowded_cat.png')
plt.show()

plt.imshow(image, cmap='Greys', origin='lower', interpolation='none', vmin=np.min(image), vmax=np.percentile(image[32:42,64:74], 97.5))
plt.colorbar()
plt.xlim((63.5,73.5))
plt.ylim((31.5,41.5))
plt.scatter(DPx, DPy, marker='+', s=2000*10**(0.4*(20-DPr)), c='b', linewidth=1.5)
for i in xrange(DPx.shape[0]):
  plt.annotate('%0.1f(%1.0f)' % (DPr[i], DPrerr[i]*10), xy=(DPx[i]+0.6, DPy[i]-0.125), fontsize=18, color='b')
plt.tight_layout()
plt.savefig('crowded_zoom.png')
plt.show()
plt.imshow(image, cmap='Greys', origin='lower', interpolation='none', vmin=np.min(image), vmax=np.percentile(image[32:42,64:74], 97.5))
plt.colorbar()
plt.xlim((63.5,73.5))
plt.ylim((31.5,41.5))
plt.scatter(HTx[HT606 < 22], HTy[HT606 < 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 < 22])), c='lime', linewidth=1.5)
plt.scatter(HTx[HT606 > 22], HTy[HT606 > 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 > 22])), c='g', linewidth=1.5)
plt.scatter(DPx, DPy,                         marker='+', s=2000*10**(0.4*(20-DPr)), c='b', linewidth=1.5)
for i in xrange(DPx.shape[0]):
  plt.annotate('%0.1f(%1.0f)' % (DPr[i], DPrerr[i]*10), xy=(DPx[i]+0.6, DPy[i]-0.125), fontsize=18, color='b')
plt.tight_layout()
plt.savefig('crowded_zoom_hst.png')
plt.show()

plt.imshow(image, cmap='Greys', origin='lower', interpolation='none', vmin=np.min(image), vmax=np.percentile(image[32:42,64:74], 97.5))
cb = plt.colorbar()
cb.set_label('ADU')
plt.xlim((63.5,73.5))
plt.ylim((31.5,41.5))
for j in xrange(PCn.shape[0]):
 label = 'Probabilistic Cataloging' if j == 0 else ''
 alpha = 1 if j == 0 else 0.1
 plt.scatter(PCx[j,0:PCn[j]], PCy[j,0:PCn[j]], marker='x', s=2000*10**(0.4*(20-adutomag(PCf[j,0:PCn[j]]))), c='r', linewidth=1.5, alpha=alpha, label=label)
plt.scatter(HTx[HT606 < 22], HTy[HT606 < 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 < 22])), c='lime', linewidth=1.5, label='HST (ground truth)')
plt.scatter(HTx[HT606 > 22], HTy[HT606 > 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 > 22])), c='g', linewidth=1.5)
plt.scatter(DPx, DPy,                         marker='1', s=2000*10**(0.4*(20-DPr)), c='aqua', linewidth=1.5, label='DAOPHOT')
#for i in xrange(DPx.shape[0]):
# plt.annotate('%0.1f(%1.0f)' % (DPr[i], DPrerr[i]*10), xy=(DPx[i]+0.6, DPy[i]-0.125), fontsize=18, color='b')
lgnd = plt.legend(scatterpoints=1)
for i in lgnd.legendHandles:
 i._sizes = [500]
plt.tight_layout()
plt.savefig('crowded_zoom_stacked.png')
plt.show()

plt.imshow(image, cmap='Greys', origin='lower', interpolation='none', vmin=np.min(image), vmax=np.percentile(image[32:42,64:74], 97.5))
plt.colorbar()
plt.scatter(HTx[HT606 < 22], HTy[HT606 < 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 < 22])), c='lime', linewidth=1.5)
plt.scatter(HTx[HT606 > 22], HTy[HT606 > 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 > 22])), c='g', linewidth=1.5)
plt.scatter(DPx, DPy,                         marker='+', s=2000*10**(0.4*(20-DPr)), c='b', linewidth=1.5)
for i in xrange(DPx.shape[0]):
 plt.annotate('%0.1f(%1.0f)' % (DPr[i], DPrerr[i]*10), xy=(DPx[i]+0.6, DPy[i]-0.125), fontsize=18, color='b')
plt.scatter(CCx, CCy, marker='x', s=2000*10**(0.4*(20-CCr)), c='r', linewidth=1.5)
for i in xrange(CCx.shape[0]):
 if CCconf[i] < 0.995:
  plt.annotate('[%1.0f%%] %0.1f(%1.0f)' % (CCconf[i]*100, CCr[i], CCrerr[i]*10), xy=(CCx[i]-0.6, CCy[i]+0.125), fontsize=18, color='r', horizontalalignment='right')
 else:
  plt.annotate('%0.1f(%1.0f)' % (CCr[i], CCrerr[i]*10), xy=(CCx[i]-0.6, CCy[i]-0.125), fontsize=18, color='r', horizontalalignment='right')
plt.xlim((63.5,73.5))
plt.ylim((31.5,41.5))
plt.tight_layout()
plt.savefig('crowded_zoom_condensed.png')
plt.show()

for j in xrange(PCn.shape[0]):
 plt.imshow(image, cmap='Greys', origin='lower', interpolation='none', vmin=np.min(image), vmax=np.percentile(image[32:42,64:74], 97.5))
 plt.colorbar()
 plt.xlim((63.5,73.5))
 plt.ylim((31.5,41.5))
 plt.scatter(HTx[HT606 < 22], HTy[HT606 < 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 < 22])), c='lime', linewidth=1.5)
 plt.scatter(HTx[HT606 > 22], HTy[HT606 > 22], marker='+', s=2000*10**(0.4*(20-HT606[HT606 > 22])), c='g', linewidth=1.5)
 plt.scatter(DPx, DPy,                         marker='+', s=2000*10**(0.4*(20-DPr)), c='b', linewidth=1.5)
 for i in xrange(DPx.shape[0]):
  plt.annotate('%0.1f(%1.0f)' % (DPr[i], DPrerr[i]*10), xy=(DPx[i]+0.6, DPy[i]-0.125), fontsize=18, color='b')
 plt.scatter(PCx[j,0:PCn[j]], PCy[j,0:PCn[j]], marker='x', s=2000*10**(0.4*(20-adutomag(PCf[j,0:PCn[j]]))), c='r', linewidth=1.5)
 for i in xrange(PCn[j]):
  plt.annotate('%0.1f' % adutomag(PCf[j,i]), xy=(PCx[j,i]-0.6, PCy[j,i]-0.125), fontsize=18, color='r', horizontalalignment='right')
 plt.tight_layout()
 plt.savefig('Frames/%0.4d.png' % j)
 plt.clf()
 print j'''
