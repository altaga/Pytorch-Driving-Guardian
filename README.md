# Pytorch Driving Guardian

Monitor completo de el estado de alerta del conductor, estado emocional y monitor inteligente del punto ciego del carro.

<img src="./Images/logo.png" width="1000">

# Table of contents

- [Pytorch Driving Guardian](#pytorch-driving-guardian)
- [Table of contents](#table-of-contents)
- [Introduction:](#introduction)
- [Problem:](#problem)
  - [Current Solutions:](#current-solutions)
  - [Theoretical Support:](#theoretical-support)
- [Solution:](#solution)
  - [Full Solution Diagrams:](#full-solution-diagrams)
  - [Drowsiness:](#drowsiness)
    - [Summary:](#summary)
    - [Neural Network:](#neural-network)
    - [Mini Demo:](#mini-demo)
      - [**Jupyter Notebook**:](#jupyter-notebook)
      - [**Demo Drowsiness**:](#demo-drowsiness)
      - [**Demo Alert**:](#demo-alert)
  - [Emotion Detection:](#emotion-detection)
    - [Summary:](#summary-1)
    - [Neural Network:](#neural-network-1)
    - [Mini Demo:](#mini-demo-1)
      - [**Jupyter Notebook**:](#jupyter-notebook-1)
      - [**Demo**:](#demo)
  - [Blind Spot:](#blind-spot)
    - [Summary:](#summary-2)
    - [Neural Network:](#neural-network-2)
    - [Mini Demo:](#mini-demo-2)
      - [**Jupyter Notebook**:](#jupyter-notebook-2)
      - [**Demo**:](#demo-1)
- [The Final Product:](#the-final-product)
    - [DEMO:](#demo)
- [Commentary:](#commentary)
  - [References:](#references)

# Introduction:

Conducir se ha vuelto una tarea tan cotidiana para el ser humano como comer, lavarse los dientes o caminar, sin embargo esta a su vez ze ha convertido es una tarea que puede consumir una gran parte de nuestro dia a dia, ademas de ser una potencialmente peligrosa si no se siguen ciertas normas de seguridad.

<img src="https://i.pinimg.com/originals/e4/26/46/e4264624281d816222229deed61c8e32.gif">

# Problem:

Hay 4 peligros muy reales y presentes a la hora de conducir, los cuales son los siguientes.

- Estar cansado, somnoliento o distraido.
  - Peligro: podria provocar un choque por quedarse dormido o distraerse con el celular.
- Estar en un estado emocional irregular como puede ser enojado o triste.
  - Esto puede generar una conduccion erratica o peligrosa, desencadenando en un gasto mucho mayor de combustible o incluso provocando un choque.
- No poder poner atencion al punto ciego del vehiculo.
  - Que al realizar un cambio de carril o dar una vuelta en alguna calle se provoque un choque o peor aun dañar a una persona.
- Tener un choque y no poder obtener una ayuda rapida.
  - Que por cualquiera de las razones anteiores o razones externas choquemos y al chocar no podamos avisar a nuestros familiares o contactos de confianza que hemos chocado y mas aun, donde.

## Current Solutions:

- Mercedes-Benz Attention Assist uses the car's engine control unit to monitor changes in steering and other driving habits and alerts the driver accordingly.

- Lexus placed a camera in the dashboard that tracks the driver's face, rather than the vehicle's behavior, and alerts the driver if his or her movements seem to indicate sleep.

- Volvo's Driver Alert Control is a lane-departure system that monitors and corrects the vehicle's position on the road, then alerts the driver if it detects any drifting between lanes.

- Saab uses two cameras in the cockpit to monitor the driver's eye movement and alerts the driver with a text message in the dash, followed by a stern audio message if he or she still seems sleepy.

As you can see these are all premium brands and there is not a single plug and play system that can work for every car. This, is our opportunity as most cars in the road are not on that price range and do not have these systems.

## Theoretical Support:

The Center for Disease Control and Prevention (CDC) says that 35% of American drivers sleep less than the recommended minimum of seven hours a day. It mainly affects attention when performing any task and in the long term, it can affect health permanently [[1]](https://medlineplus.gov/healthysleep.html).

<img src="https://www.personalcarephysicians.com/wp-content/uploads/2017/04/sleep-chart.png" width="1000">

According to a report by the WHO (World Health Organization) [[2]](http://www.euro.who.int/__data/assets/pdf_file/0008/114101/E84683.pdf), falling asleep while driving is one of the leading causes of traffic accidents. Up to 24% of accidents are caused by falling asleep, and according to the DMV USA (Department of Motor Vehicles) [[3]](https://dmv.ny.gov/press-release/press-release-03-09-2018) and NHTSA (National Highway traffic safety administration) [[4]](https://www.nhtsa.gov/risky-driving/drowsy-driving), 20% of accidents are related to drowsiness, being at the same level as accidents due to alcohol consumption with sometimes even worse consequences than those.

<img src="https://media2.giphy.com/media/PtrhzZJhbEBm8/giphy.gif" width="1000">

Also, the NHTSA mentions that being angry or in an altered state of mind can lead to more dangerous and aggressive driving [[5]](https://www.nhtsa.gov/risky-driving/speeding), endangering the life of the driver due to these psychological disorders.

<img src="https://i.ibb.co/YcWYJNw/tenor-1.gif" width="1000">

# Solution:

Contruimos un prototipo el cual es capaz de realizar estos 3 monitoreos de forma fiable y ademas de facil instalacion en cualquier vehiculo.

<img src="./Images/logo.png" width="1000">

Este POC usa como computadora principal una Jetson Nano 4gb en modo 5W para mantener un consumo bajo para su uso continuo en un vehiculo. La Jetson Nano es una mini computadora muy parecida a la RaspberryPi, con la diferencia que esta tiene una GPU Dedicada habilitada con CUDA, con el fin de ejecutar sobre la GPU los modelos de AI de Pytorch.

<img src="./Images/Jetson.jpg">

Para visualizar los resultados se uso un M5core2, el cual es un dispositivo IoT con una pantalla capaz de mostrar los datos a travez de MQTT.

<img src="./Images/m5core.jpg">

## Full Solution Diagrams:

This is the connection diagram of the system:

<img src="./Images/diagram.jpg" width="1000">

Ya todo el device montado en el auto se veria asi.

<img src="./Images/POC.jpg">

## Drowsiness:

### Summary:

The function of this model is to make a detection of distraction or closed eyes of the driver for more than 2 seconds (Drowsiness) or he is distracted from the road (for example, looking at the cell phone).

[More Info](https://github.com/altaga/Pytorch-Driving-Guardian/tree/main/Hardware%20Code/Jetson%20Code/Drowsiness/README.md)

### Neural Network:

La red neuronal que ocupamos para este problema es una red neuronal convolucional, sin embargo como parte de optimizar esta red con las increibles herramientas de Pytorch.

<img src="./Images/nnDrow.png">

Layers:

- Input Layer: Esta layer tiene una entrada de (24,24,1), recibiendo los datos de una imagen de 24 px de alto y 24px de largo en escala de grises.
- Conv2D: Layer convolucional para la generacion de filtros de las imagenes de entrada.
- BatchNorm2D: Ayuda a la capa convolucional a normalizar los valores despues de esa capa y ayuda a la red a acelerar su convergencia en el entrenamiento.
  - https://arxiv.org/abs/1502.03167
- ReLu: con esta capa eliminamos las activaciones negativas despues de cada normalizacion.
- ResidualBlock: Este tipo de bloque mejora el rendimiento de la red haciendo que cada uno aprenda aun mas sobre los datos que buscamos analizar evitando la degradacion del performance al agregar aun mas bloques.
  - https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278

Dato Curioso: El uso de Residual Blocks tiene como funcion evitar las perdidas de una CNN mientras la red crece, un problema similar a las redes RNN para NLP, las cuales tiene como solucion las Redes Neuronales Transformer.

### Mini Demo:

#### **Jupyter Notebook**:

Si quieres probar la funcion del modelo, he realizado un Jupyter Notebook ya con el codigo listo para funcionar.

Link: PENDING

#### **Demo Drowsiness**:

Video: Click on the image
[![Car](./Images/logo.png)](https://youtu.be/Ttvc87rZZBw)
Sorry github does not allow embed videos.

#### **Demo Alert**:

Video: Click on the image
[![Car](./Images/logo.png)](https://youtu.be/TtqtQ1KRBL4)
Sorry github does not allow embed videos.

## Emotion Detection:

### Summary:

The function of this model is to detect the driver's emotions at all times and through musical responses (songs) try to correct the driver's mental state, in order to keep him neutral or in a good mood while driving, thus reducing the risk of accidents.

[More Info](https://github.com/altaga/Pytorch-Driving-Guardian/tree/main/Hardware%20Code/Jetson%20Code/Emotion%20detection/README.md)

### Neural Network:

Se utilizo la misma red neuronal que en el caso anterior ya que el problema tambien requiere el uso de una red neuronal convolucional.

<img src="./Images/nnEmotion.png">

Layers:

- Input Layer: Esta layer tiene una entrada de (24,24,1), recibiendo los datos de una imagen de 48 px de alto y 48px de largo en escala de grises.
- Conv2D: Layer convolucional para la generacion de filtros de las imagenes de entrada.
- BatchNorm2D: Ayuda a la capa convolucional a normalizar los valores despues de esa capa y ayuda a la red a acelerar su convergencia en el entrenamiento.
  - https://arxiv.org/abs/1502.03167
- ReLu: con esta capa eliminamos las activaciones negativas despues de cada normalizacion.
- ResidualBlock: Este tipo de bloque mejora el rendimiento de la red haciendo que cada uno aprenda aun mas sobre los datos que buscamos analizar evitando la degradacion del performance al agregar aun mas bloques.
  - https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278

### Mini Demo:

#### **Jupyter Notebook**:

Si quieres probar la funcion del modelo, he realizado un Jupyter Notebook ya con el codigo listo para funcionar.

Link: PENDING

#### **Demo**:

Video: Click on the image
[![Car](./Images/logo.png)](https://www.youtube.com/watch?v=-vTpf9aUerA)
Sorry github does not allow embed videos.

## Blind Spot:

### Summary:

The function of this model is to detect objects that are less than 3 meters from the car at the blind spot.

[More Info](https://github.com/altaga/Pytorch-Driving-Guardian/tree/main/Hardware%20Code/Jetson%20Code/YoloV3/README.md)

### Neural Network:

Para detectar multiples objetos en una imagen como lo son personas, autos o animales. Se decidio que lo mas eficente era utilizar una red ya pre-entrenada y con la capacidad de realizar esta tarea de forma eficiente, por lo tanto decidimos utilizar una Darknet, en especifico la YoloV3.

<img src="https://i.stack.imgur.com/js9wN.png">

Layers:

- Input Layer: Esta layer tiene una entrada de (416,416,3), recibiendo los datos de una imagen de 416 px de alto y 416px de largo a color.
- ConvolutionDownsampling: Esta capa tiene la funcion de realizar un pooling de la imagen y empezar a genera los filtros de la imagen.
- Dense Connection: Esta capa es una red de neuronas normales conectadas, como cualquier capa densa en una red neuronal.
- Spatial Pyramid Pooling: Given an 2D input Tensor, Temporal Pyramid Pooling divides the input in x stripes which extend through the height of the image and width of roughly (input_width / x). These stripes are then each pooled with max- or avg-pooling to calculate the output.
  - https://github.com/revidee/pytorch-pyramid-pooling
- Object Detection: Esta capa tiene como finalidad terminar de determinar los objetos que estan siendo observados en la imagen.

### Mini Demo:

#### **Jupyter Notebook**:

Si quieres probar la funcion del modelo, he realizado un Jupyter Notebook ya con el codigo listo para funcionar.

Link: PENDING

#### **Demo**:

Video: Click on the image
[![Car](./Images/logo.png)](https://youtu.be/yygXL4Zh7i4)
Sorry github does not allow embed videos.

# The Final Product:

Product installed inside the car:

<img src="./Images/d3.jpg" width="800">
<img src="./Images/d1.jpg" width="800"> 

Notifications:

<img src="./Images/message.jpg" width="600">

### Epic DEMO:

Video: Click on the image
[![Car](./Images/logo.png)](pending)
Sorry github does not allow embed videos.

# Commentary:

I would consider the product finished as we only need a little of additional touches in the industrial engineering side of things for it to be a commercial product. Well and also a bit on the Electrical engineering perhaps to use only the components we need. That being said this functions as an upgrade from a project that a couple friends and myself are developing and It was ideal for me to use as a springboard and develop the idea much more. This one has the potential of becoming a commercially available option regarding Smart cities as the transition to autonomous or even smart vehicles will take a while in most cities.

That middle ground between the Analog, primarily mechanical-based private transports to a more "Smart" vehicle is a huge opportunity as the transition will take several years and most people are not able to afford it. Thank you for reading.

## References:

Links:

1. https://medlineplus.gov/healthysleep.html

2. http://www.euro.who.int/__data/assets/pdf_file/0008/114101/E84683.pdf

3. https://dmv.ny.gov/press-release/press-release-03-09-2018

4. https://www.nhtsa.gov/risky-driving/drowsy-driving

5. https://www.nhtsa.gov/risky-driving/speeding

