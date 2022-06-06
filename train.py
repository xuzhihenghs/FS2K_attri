from faceAttr_trainer import Classifier_Trainer
#import config as cfg


trainer = Classifier_Trainer(32, 32, 1e-5, 'VGG16')
trainer.fit()
