import os
import pytorch_lightning as pl

chpt_callback = pl.callbacks.ModelCheckpoint(filepath='trained_models/',
                                             save_top_k=1,
                                             verbose=False,
                                             monitor='val_loss',
                                             mode='min',
                                             prefix='')
