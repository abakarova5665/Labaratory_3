# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнила:
- Абакарова Кистаман Умарасхабовна
- ХС21
Отметка о выполнении заданий:

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы: познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.


## Задание 1
Ход работы:

- Установка MLAgents и pytorch
- Создание Prefab с сферой, плоскостью и кубом.



На первом этапе было необходимо создать новый проект Unity, создать скрипт RollerAgent.cs и привязать его к сфере.

![sphere](https://user-images.githubusercontent.com/48391156/197057045-0e096bfa-e3c2-45ec-b960-3c250de26fab.png)


Объекту «сфера» добавиkf компоненты Rigidbody, Decision Requester,
Behavior Parameters и настроила их 
![RollerAgent](https://user-images.githubusercontent.com/48391156/197057528-d1fb6131-7a02-49e3-ad30-fb64757022eb.png)

Обучение MLAgent в конфигурации rollerball_config.yaml на 9 копиях ранее созданного Prefab

```
mlagents-learn rollerball_config.yaml --run-id=RollerBall --force
```
![ml_agent](https://user-images.githubusercontent.com/48391156/197058832-adbd2803-a05e-4e59-bc2a-5ab65e1da5bc.png)

Рисунок 3 

Итог 
![lab3](https://user-images.githubusercontent.com/48391156/197063956-d291e51b-0819-4619-9997-893eab0676ff.png)

Рисунок 4

- Проверила работу MLAgent. Итогом обучения стал показатель средней награды в 0.963.

![ml_ag2](https://user-images.githubusercontent.com/48391156/197065297-a5216f82-5b94-4907-88ed-2b8c8f8244d9.png)


## Задание 2

- Компонент Decision Requester отвечает за запрос решений для агента через равные промежутки времени. Для этого вызывается функция RequestDecision у класса Agent, от которого наследуется созданный скрипт.
- Decision Period определяет частоту шагов обучение. Раз в заданное количество шагов, MLAgent будет запрашивать решение. 
- Behavior Parametrs -  компонент определяющий принятие решений объектом.
- Behavior Name - имя поведения, которое используется в виде базового имени.
- Behavior Type - определяет, какой тип поведения будет использовать агент. 
    
    - *Default* - агент будет использовать удаленный процесс обучения, запущенный через python. 
    - *InferenceOnly* - агент всегда будет использовать предоставленную моделью нейронной сети. 
    - *HeuristicOnly* - всегда используется эвристический метод.
- Vector Observation - это вектор чисел с плавающей запятой, которые содержат релевантную информацию для принятия агентом решений. Вектор заполняется в функции CollectObservations.

```cs
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(FirstTarget.transform.localPosition);
        sensor.AddObservation(SecondTarget.transform.localPosition);
        sensor.AddObservation(isFirstTargetActive);
        sensor.AddObservation(isSecondTargetActive);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
```

- Actions - MLAgent выдаются инструкции в форме действий, они делятся на два типа: 
1) непрерывные 
2) дискретные

- Branch Sizes - определяет массив размеров ветвей для дискретных действий. 

- Continuous Actions определяет количество доступных непрерывных действий. 

- Алгоритм обучения пробует разные значения ActionBuffers и наблюдает за влиянием накопленных вознаграждений в течении всех итераций обучения. Действия для MLAgent описываются в функции OnActionRecieved(). 

```cs
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);
    ...
```

Конфигурация rollerball_config.yaml используется для изменения параметров обучения модели.

```yaml
# Набор конфигураций для определяющий поведение агентов
behaviors:
  RollerBall:
    # Определения алгоритма обучения. В данном случае используется Proximal Policy Optimization - это алгоритм обучения с подкреплением
    trainer_type: ppo
    # Гиперпараметры модели
    hyperparameters:
      # Количество опыта на каждой итерации градиентного спуска.
      batch_size: 10
      #  Количество опыта, необходимое для начала обновления модели политики
      buffer_size: 100
      # Начальная скорость обучения для градиентного спуска. Соответствует силе каждого шага обновления градиентного спуска.
      learning_rate: 3.0e-4
      # Сила регуляризации энтропии - необходимо для рандомизации политики. Необходимо для исследования пространства действий во время обучения. 
      beta: 5.0e-4
      # Параметр допустимого расхождения между старой и новой политикой. Влияет на скорость обучения и стабильность обновлений политики.
      epsilon: 0.2
      #  Процент использования текущей политики для предсказаний
      lambd: 0.99
      # Количество проходов, которые необходимо выполнить через буфер опыта при выполнении оптимизации градиентного спуска. Количество эпох перед оптимизацией градиентного спуска
      num_epoch: 3
      # Определяет изменение скорости обучение во времени.
      learning_rate_schedule: linear
    # Настройки нейронной сети
    network_settings:
      # Применяется ли нормализация к входным данным векторного наблюдения. 
      normalize: false
      # Количество нейронов в скрытых слоях нейронной сети
      hidden_units: 128
      # Количество скрытых слоев в нейронной сети после приема входящих данных.
      num_layers: 2
    # Параметры внешних и внутренних сигналов вознаграждения
    reward_signals:
      extrinsic:
        # Коэффициент дисконтирования будущих вознаграждений, поступающих от окружающей среды
        gamma: 0.99
        # Мультипликатор вознаграждения окружающей среды
        strength: 1.0
    # Количество шагов необходимого для завершения обучения
    max_steps: 500000
    # Количество шагов необходимого для добавления опыта в буфер.
    time_horizon: 64
    # Количество опыта необходимое для отображения статистики обучения
    summary_freq: 10000
    # Использование многопоточности
    threaded: true
```



## Выводы
Узнала, что такое ML-агенты, и, более или менее, поняла что они умеют. Научилась реализовывать систему машинного обучения в связке Python - Google Sheets - Unity. Разобрала строчки файла конфигурации нейронной сети и немногот изучила информацию о компанентах Decision Requester, Behavior Parameters.


| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
