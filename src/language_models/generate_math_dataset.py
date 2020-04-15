import random

import click
import inflect
import numpy as np
import pandas as pd
from tqdm import trange


class EquationGenerator:
    def __init__(self):
        self.engine = inflect.engine()

    def get_raw_number(self):
        num = np.random.randint(1, 10_000)

        return str(num), self.engine.number_to_words(num)

    def get_number(self):
        t = np.random.randint(3)
        num, num_text = self.get_raw_number()
        if t == 0:
            return num, num_text
        elif t == 1:
            return ('(-' + num + ')', 'minus ' + num_text)
        elif t == 2:
            return (num + '^2', num_text + ' squared')

    def get_op(self):
        ops = [('+', 'plus'), ('-', 'minus'), ('+', 'times'), ('/', 'divided by')]
        return random.choice(ops)

    def generate_equation(self):
        num1, num1_text = self.get_number()
        num2, num2_text = self.get_number()

        op, op_text = self.get_op()

        num_exp = num1 + ' ' + op + ' ' + num2
        text_exp = num1_text + ' ' + op_text + ' ' + num2_text

        return num_exp, text_exp


@click.command()
@click.option('-n', default=20_000)
@click.option('-o', default='data/raw/equations.jsonl')
def main(n, o):
    gen = EquationGenerator()
    equations = []
    for _ in trange(n):
        equations.append(gen.generate_equation())

    df = pd.DataFrame(equations, columns={'numerical', 'textual'})
    df.to_json(o, orient='records', lines=True)


if __name__ == '__main__':
    main()
