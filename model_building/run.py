from generators_factory import GeneratorsFactory

def makeData(''):


def main():

    factory = GeneratorsFactory('parameters.ini', 123)
    gen = factory.build()
    expconf = gen.generate_experiment_configurations()
    for exp in expconf:
        exp.train()


if __name__ == '__main__':
    main()
