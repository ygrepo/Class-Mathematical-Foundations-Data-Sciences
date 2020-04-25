# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:39:11 2017
Functions for plotting images stored in vectors.

@author: carlos
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

"""
Plots a vector on the screen using grayscale.  The parameter
image_shape is used to reshape the vector into a 2-dimensional image.
The image is also saved to a PDF file title.pdf.
"""
def plot_image(image,title,image_shape=(64,64)):
    plt.figure()
    plt.imshow(image.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=image.min(), vmax=image.max())
    plt.title(title)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(title + '.pdf',bbox_inches='tight')
    plt.show()

"""
Plots a list of vectors on the screen in a grid with n_col columns
and n_row rows.  The resulting grid is saved in the file title.pdf.
If bycol is True, then the images are laid out by columns instead of by rows.
Supports optional row and column titles.
"""
def plot_image_grid(images, title, image_shape=(64,64),n_col=5, n_row=2, bycol=0, row_titles=None,col_titles=None):
    fig,axes = plt.subplots(nrows=n_row,ncols=n_col,figsize=(2. * n_col, 2.26 * n_row))
    for i, comp in enumerate(images):
        row,col = reversed(divmod(i,n_row)) if bycol else divmod(i,n_col)       
        cax = axes[row,col] if n_row>1 and n_col>1 else (axes[row] if row > 1 else axes[col])
        cax.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=comp.min(), vmax=comp.max())
        cax.set_xticks(())
        cax.set_yticks(())
    if row_titles is not None :
        for ax,row in zip(axes[:,0],row_titles) :
            ax.set_ylabel(row,size='large')
    if col_titles is not None :
        for ax,col in zip(axes[0,:] if n_row > 1 else axes,col_titles) :
            ax.set_title(col)
    
    #fig.suptitle(title)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(title + '.pdf',bbox_inches='tight')
    plt.show()

"""
Plots the 2d fourier transforms of 
a list of vectors on the screen in a grid with n_col columns
and n_row rows.  The resulting grid is saved in the file title.pdf.
If bycol is True, then the images are laid out by columns instead of by rows.
Supports optional row and column titles.
"""
def plot_fft_image_grid(images, title, image_shape=(64,64),n_col=5, n_row=2, bycol=0, row_titles=None,col_titles=None):
    fig,axes = plt.subplots(nrows=n_row,ncols=n_col,figsize=(2. * n_col, 2.26 * n_row))
    m = np.amin([np.amin(np.abs(np.fft.fft2(im))) for im in images])
    M = np.amax([np.amax(np.abs(np.fft.fft2(im))) for im in images])
    for i, comp in enumerate(images):
        row,col = reversed(divmod(i,n_row)) if bycol else divmod(i,n_col)       
        cax = axes[row,col] if n_row>1 and n_col>1 else (axes[row] if row > 1 else axes[col])
        fftim = np.fft.fftshift(np.fft.fft2(comp.reshape(image_shape)))
        ax = cax.imshow(np.abs(fftim),norm=LogNorm(vmin=m+1e-9,vmax=M))
        cax.set_xticks(())
        cax.set_yticks(())
        fig.colorbar(ax,ax=cax)
    if row_titles is not None :
        for ax,row in zip(axes[:,0],row_titles) :
            ax.set_ylabel(row,size='large')
    if col_titles is not None :
        for ax,col in zip(axes[0,:] if n_row > 1 else axes,col_titles) :
            ax.set_title(col)
            
    #fig.suptitle(title)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(title +'.pdf',bbox_inches='tight')
    plt.show()
