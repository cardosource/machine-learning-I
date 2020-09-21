from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules.gaussianlayer import GaussianLayer
from pybrain.datasets import  SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


artefato ={'entrada1': (0, 0),
           'entrada2': (0, 1),
           'entrada3': (1, 0),
           'entrada4': (1, 1),

           'padrao1': (0,),
           'padrao2': (1,),
           'padrao3': (1,),
           'padrao4': (0,)
           }


rede =buildNetwork(2,3,1,outclass=GaussianLayer)
for i in range(1, 5):
    entrada = f'entrada{i}'
    padrao = f'padrao{i}'
    print(artefato[entrada], " - ", artefato[padrao])

    base = SupervisedDataSet(2, 1)
    base.addSample(artefato[entrada], artefato[padrao])
    treinamento = BackpropTrainer(rede, dataset=base, learningrate=0.01, momentum=0.06)

    for interacao in range(500):
        erro = treinamento.train()
        print("erro", erro)

print(rede.activate([0, 0]))
print(rede.activate([1, 0]))
print(rede.activate([0, 1]))
print(rede.activate([1, 0]))

