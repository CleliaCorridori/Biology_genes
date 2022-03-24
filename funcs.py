import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm
import matplotlib
import numpy.linalg as la

def Pearson_info(matx, det=False):
    corr_matx = np.corrcoef(matx)
    
    if det==True:
        determ = np.linalg.det(corr_matx)
        print("determinant: ", determ)
#         print("determinant: ", np.round(determ,6))
    
    return(corr_matx)

def fig_matx(matx, ax_names, titl):
    x_pos = np.arange(0,np.shape(matx)[0])
    
    plt.figure(figsize=(12,10))
    plt.imshow(matx)
    plt.xticks(x_pos, labels=ax_names, rotation='vertical')
    plt.yticks(x_pos, ax_names)
    plt.title(titl)
    plt.colorbar()
    
def fig_distribution(matx, titl):
    vect =  np.squeeze(np.reshape(matx, (np.shape(matx)[0]*np.shape(matx)[1],1)))
    
    plt.figure(figsize=(8,5))
    counts, bins, patches = plt.hist(vect, 30, range=(-1,1), density=True, color="Navy")
    plt.grid()
    plt.title(titl, fontsize=14)
    plt.xlabel("Pearson Correlation Coefficients", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.show()
    
    centroids = (bins[1:] + bins[:-1]) / 2
    return(counts, bins, patches, centroids)

def fig_distr_int(matx, thr, text="Interactions values", Print=False):
    lin_vect = np.squeeze(np.reshape(matx, (np.shape(matx)[0]*np.shape(matx)[1],1)))
    plt.figure(figsize=(12,7))
    plt.hist(lin_vect, 35, density=True, color="Navy")
    plt.grid()
    plt.title(text, fontsize=14)
    plt.xlabel("Interaction Coefficients", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
#     plt.axvline(np.mean(matx)+np.std(matx), color="red", label="+"+np.str(np.round(np.mean(matx)+np.std(matx),0)) )
#     plt.axvline(np.mean(matx)-np.std(matx), color="red", label="-"+np.str(np.round(np.mean(matx)-np.std(matx),0)) )
    plt.legend(fontsize="14")
    if Print==True:
        plt.savefig('Distr.pdf', transparent = False, bbox_inches='tight')
    plt.show()
      
    
def eigval_analysis(matx, thr = 0.0001, info=True, fig=True):
    eigval, eigvect = np.linalg.eig(matx)
    if info==True:
        print("# eigvals below thr =",thr, ": ", np.shape(np.real(eigval[np.abs(eigval)<thr])))
        
    if fig==True: 
        plt.hist(np.real(np.abs(eigval)), 30, density=False, color="Navy")
        plt.grid()
        plt.title("2i+LIF", fontsize=14)
        plt.xlabel("Eigenvalues", fontsize=14)
        plt.ylabel("Counts", fontsize=14)
        plt.show()
    return(np.real(eigval), np.real(eigvect))
    

def spectr_decomposition(eigval, eigvect, thr = 0.01, info=False):
#     eigval_sel = np.unique(eigval[np.abs(eigval)>=thr])
    eigval_sel = eigval[np.abs(eigval)>=thr]
    matx_int = np.zeros((np.shape(eigvect)[0],np.shape(eigvect)[0]))
    
    for kk in range(0,len(eigval_sel)):
        for ii in range(np.shape(eigvect)[0]):
            for jj in range(np.shape(eigvect)[0]):
                matx_int[ii,jj] += (eigval_sel[kk])**(-1) * eigvect[ii,kk] * (eigvect)[jj,kk]
    return(matx_int)

def sp_dec(matx, thr = 0.0001):#############################################################
    eigval, eigvect = la.eig(matx)
    idx_to_save = []
    for kk in range(len(eigval)):
        if eigval[kk] < thr:
            idx_to_save.append(kk)
    for jj in range(len(idx_to_save)):
        eigval = np.delete(eigval,jj)
        eigvect = np.delete(eigvect, jj, 1)
        eigvect = np.delete(eigvect, jj, 0)
    inv_matx = np.dot(eigvect,np.dot(np.diag(1/eigval), la.inv(eigvect)))
    return(inv_matx)


def fig_matx_int(matx, ax_names, titl, thr=2, lintsh=1):
    x_pos = np.arange(0,np.shape(matx)[0])
    
    masked_array = np.ma.masked_where(np.abs(matx) <= thr, matx)
#     cmap = matplotlib.cm.plasma
    cmap = matplotlib.cm.coolwarm
#     cmap = "coolwarm_nocenter_log"
    cmap.set_bad(color='white')
    limit = max(abs(np.min(matx)), abs(np.max(matx)) )
    plt.figure(figsize=(12,10))
    plt.imshow(masked_array, norm=colors.SymLogNorm(lintsh), cmap=cmap)
#     plt.imshow(masked_array, cmap=cmap)
    plt.xticks(x_pos, labels=ax_names, rotation='vertical')
    plt.yticks(x_pos, ax_names)
    plt.title(titl)
    plt.colorbar()
    plt.show()
    

def fig_matx_int_text(matx, ax_names, titl, thr=2, lintsh=1, log=True, info=True):
    x_pos = np.arange(0,np.shape(matx)[0])
    
    masked_array = np.ma.masked_where(np.abs(matx) <= thr, matx)
    cmap = matplotlib.cm.coolwarm
    cmap.set_bad(color='white')
    limit = max(abs(np.min(matx)), abs(np.max(matx)) )
    fig, ax = plt.subplots(figsize=(20,16))
    if log==True:
        plt.imshow(masked_array, norm=colors.SymLogNorm(lintsh), cmap=cmap)
    else:
        plt.imshow(masked_array, cmap=cmap)

    plt.xticks(x_pos, labels=ax_names, rotation='vertical')
    plt.yticks(x_pos, ax_names)
    plt.title(titl)
    plt.colorbar()
    # Loop over data dimensions and create text annotations.
    for i in range(np.shape(matx)[0]):
        for j in range(np.shape(matx)[1]):
            text = ax.text(j, i, np.round(matx[i, j],2),
                           ha="center", va="center", color="w")
    if info==True:
        print(("min", np.min(matx), "max", np.max(matx)))
        plt.savefig('Prova.pdf', transparent = False, bbox_inches='tight')
    plt.show()

    
def ind_val(matx, num=1, MM=True, names=True):
    vect_sorted = np.sort(np.squeeze(np.reshape(matx, (np.shape(matx)[0]*np.shape(matx)[1],1))))
    if MM==True:
        out = np.unique(vect_sorted)[0:num]
    if MM==False:
        out = np.unique(vect_sorted)[-num:]
        
    rows=[]
    cols=[]
    for ii in range(0, len(out)):
        rows.append(np.where(matx==out[ii])[0][0])
        cols.append(np.where(matx==out[ii])[1][0])
    
    genes_couples = []
    if names==True:
        for jj in range(0, len(rows)):
            genes_couples.append([rows[jj],cols[jj]])
        return(genes_couples)
    else:
        return(np.array(rows), np.array(cols))
    
def ind_val_abs(matx, num=1, names=True):
    vect_sorted = np.sort(np.abs(np.squeeze(np.reshape(matx, (np.shape(matx)[0]*np.shape(matx)[1],1)))))
    out = np.unique(vect_sorted)[0:num]
        
    rows=[]
    cols=[]
    for ii in range(0, len(out)):
        rows.append(np.where(np.abs(matx)==out[ii])[0][0])
        cols.append(np.where(np.abs(matx)==out[ii])[1][0])
    
    genes_couples = []
    if names==True:
        for jj in range(0, len(rows)):
            genes_couples.append([rows[jj],cols[jj]])
        return(genes_couples)
    else:
        return(np.array(rows), np.array(cols))

def gene_list(rows, cols, gene_names):
    genes = []
    for ii in range(0,len(rows)):
        genes.append([gene_names[rows[ii]], gene_names[cols[ii]]])
    return genes
    
