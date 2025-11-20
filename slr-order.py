# ==========================
#  Simple SLR(1) parser generator
#  Ejemplo con gramática:
#     S -> A A
#     A -> a A | b
#  Imprime ACTION y GOTO usando rich.Table
# ==========================

from collections import defaultdict
from rich.console import Console
from rich.table import Table

EPSILON = 'ε'
ENDMARK = '$'


class Grammar:
    def __init__(self, productions, start_symbol, terminals=None):
        """
        productions: dict[str, list[list[str]]]
            {
              'S': [['A', 'A']],
              'A': [['a', 'A'], ['b']]
            }

        start_symbol: símbolo inicial (por ejemplo 'S')
        terminals: conjunto explícito de terminales (por ejemplo {'a', 'b'})
        """

        # Copia explícita preservando el orden del diccionario del usuario
        self.productions = {
            A: [list(rhs) for rhs in rhs_list]
            for A, rhs_list in productions.items()
        }
        self.start_symbol = start_symbol

        # Orden de no terminales definidos por el usuario (S, E, T, ...)
        self._user_nonterminals_order = list(self.productions.keys())

        self.nonterminals = set(self.productions.keys())

        # Si no pasan terminales explícitos, se infieren
        if terminals is None:
            syms = set()
            for rhs_list in self.productions.values():
                for rhs in rhs_list:
                    syms.update(rhs)
            self.terminals = (syms - self.nonterminals) - {EPSILON}
        else:
            self.terminals = set(terminals)

        # símbolo inicial aumentado S'
        self.aug_start = self.start_symbol + "'"
        if self.aug_start in self.nonterminals:
            raise ValueError("Conflicto con símbolo inicial aumentado")

        # añadimos producción S' -> S (queremos que sea la #0)
        self.productions[self.aug_start] = [[self.start_symbol]]
        self.nonterminals.add(self.aug_start)

        # -------------------------
        # Orden de producciones
        # -------------------------
        # Queremos:
        #   0: S' -> S
        #   1..n: resto en el orden en que las definió el usuario
        self.prod_order = {}
        idx = 0

        # 1) Primero la producción aumentada S' -> S
        for rhs in self.productions[self.aug_start]:
            key = (self.aug_start, tuple(rhs))
            if key not in self.prod_order:
                self.prod_order[key] = idx
                idx += 1

        # 2) Luego todas las producciones del usuario, en su orden original
        for A in self._user_nonterminals_order:
            for rhs in self.productions[A]:
                key = (A, tuple(rhs))
                if key not in self.prod_order:
                    self.prod_order[key] = idx
                    idx += 1

        # -------------------------
        # Orden de símbolos (para ordenar ítems)
        # -------------------------
        # Orden secuencial según van apareciendo:
        #   empezando por S', luego S, luego resto de no terminales y símbolos
        #   al recorrer las producciones en orden.
        self.symbol_rank = {}
        sym_idx = 0

        def register_symbol(sym):
            nonlocal sym_idx
            if sym not in self.symbol_rank:
                self.symbol_rank[sym] = sym_idx
                sym_idx += 1

        # Aseguramos primero S' y luego S
        register_symbol(self.aug_start)
        register_symbol(self.start_symbol)

        # Luego, en el orden de los no terminales del usuario, sus LHS y RHS
        for A in self._user_nonterminals_order:
            register_symbol(A)
            for rhs in self.productions[A]:
                for X in rhs:
                    register_symbol(X)

    # -------------------------
    # FIRST
    # -------------------------
    def compute_first_sets(self):
        first = {A: set() for A in self.nonterminals}

        # FIRST de terminal es él mismo
        for t in self.terminals:
            first[t] = {t}
        first[EPSILON] = {EPSILON}

        changed = True
        while changed:
            changed = False
            for A in self.nonterminals:
                for rhs in self.productions[A]:
                    # A -> X1 X2 ... Xn
                    i = 0
                    add_epsilon = True
                    while i < len(rhs):
                        X = rhs[i]

                        if X in self.terminals:
                            if X not in first[A]:
                                first[A].add(X)
                                changed = True
                            add_epsilon = False
                            break

                        # X es no terminal
                        for a in first[X]:
                            if a != EPSILON and a not in first[A]:
                                first[A].add(a)
                                changed = True

                        if EPSILON in first[X]:
                            i += 1
                        else:
                            add_epsilon = False
                            break

                    if add_epsilon:
                        if EPSILON not in first[A]:
                            first[A].add(EPSILON)
                            changed = True

        self.first = first
        return first

    def first_of_string(self, symbols):
        """FIRST de una secuencia X1 X2 ... Xn"""
        if not symbols:
            return {EPSILON}
        result = set()
        i = 0
        while i < len(symbols):
            X = symbols[i]
            if X in self.terminals:
                result.add(X)
                return result
            # no terminal
            result |= (self.first[X] - {EPSILON})
            if EPSILON in self.first[X]:
                i += 1
            else:
                return result
        result.add(EPSILON)
        return result

    # -------------------------
    # FOLLOW
    # -------------------------
    def compute_follow_sets(self):
        if not hasattr(self, "first"):
            self.compute_first_sets()

        follow = {A: set() for A in self.nonterminals}
        # por convenio, $ en FOLLOW(S)
        follow[self.start_symbol].add(ENDMARK)

        changed = True
        while changed:
            changed = False
            for A in self.nonterminals:
                for rhs in self.productions[A]:
                    # recorre A -> X1 X2 ... Xn
                    for i, B in enumerate(rhs):
                        if B in self.nonterminals:
                            beta = rhs[i + 1:]
                            first_beta = self.first_of_string(beta)

                            # FIRST(beta) sin epsilon a FOLLOW(B)
                            for a in first_beta - {EPSILON}:
                                if a not in follow[B]:
                                    follow[B].add(a)
                                    changed = True

                            # si beta puede epsilon, FOLLOW(A) ⊆ FOLLOW(B)
                            if EPSILON in first_beta or not beta:
                                for a in follow[A]:
                                    if a not in follow[B]:
                                        follow[B].add(a)
                                        changed = True

        self.follow = follow
        return follow

    # -------------------------
    # LR(0) items y colección canónica
    # -------------------------
    def closure(self, items):
        """
        items: set of (A, rhs_tuple, dot_pos)
        """
        closure_set = set(items)
        changed = True
        while changed:
            changed = False
            new_items = set()
            for (A, rhs, dot) in closure_set:
                if dot < len(rhs):
                    B = rhs[dot]
                    if B in self.nonterminals:
                        for prod in self.productions[B]:
                            item = (B, tuple(prod), 0)
                            if item not in closure_set:
                                new_items.add(item)
            if new_items:
                closure_set |= new_items
                changed = True
        return frozenset(closure_set)

    def goto(self, items, symbol):
        moved = set()
        for (A, rhs, dot) in items:
            if dot < len(rhs) and rhs[dot] == symbol:
                moved.add((A, rhs, dot + 1))
        if not moved:
            return frozenset()
        return self.closure(moved)

    def build_lr0_collection(self):
        start_item = (self.aug_start,
                      tuple(self.productions[self.aug_start][0]),
                      0)
        I0 = self.closure({start_item})
        C = [I0]
        transitions = {}  # (state_index, symbol) -> state_index

        symbols = list(self.nonterminals | self.terminals)

        changed = True
        while changed:
            changed = False
            for i, I in enumerate(C):
                for X in symbols:
                    J = self.goto(I, X)
                    if J and J not in C:
                        C.append(J)
                        changed = True
                    if J:
                        j_index = C.index(J)
                        transitions[(i, X)] = j_index

        self.lr0_states = C
        self.lr0_transitions = transitions
        return C, transitions

    # -------------------------
    # Construcción de tabla SLR(1)
    # -------------------------
    def build_slr_table(self):
        if not hasattr(self, "follow"):
            self.compute_follow_sets()
        if not hasattr(self, "lr0_states"):
            self.build_lr0_collection()

        num_states = len(self.lr0_states)
        action = {i: {} for i in range(num_states)}
        goto_table = {i: {} for i in range(num_states)}

        # shifts y goto
        for (i, X), j in self.lr0_transitions.items():
            if X in self.terminals:
                if X in action[i] and action[i][X] != ("shift", j):
                    print(f"Conflicto en ACTION[{i}, {X}]")
                action[i][X] = ("shift", j)
            elif X in self.nonterminals:
                goto_table[i][X] = j

        # reducciones y aceptación
        for i, I in enumerate(self.lr0_states):
            for (A, rhs, dot) in I:
                if dot == len(rhs):  # A -> α ·
                    if A == self.aug_start:
                        action[i][ENDMARK] = ("accept", None)
                    else:
                        for a in self.follow[A]:
                            if a in action[i] and action[i][a] != ("reduce", (A, rhs)):
                                print(f"Conflicto en ACTION[{i}, {a}]: {action[i][a]} vs reduce {A}->{rhs}")
                            action[i][a] = ("reduce", (A, rhs))

        self.action = action
        self.goto_table = goto_table
        return action, goto_table

    # -------------------------
    # Parser SLR(1)
    # -------------------------
    def parse(self, tokens):
        """
        tokens: lista de terminales, por ejemplo ['a','a','b','b']
        Devuelve True si acepta, False si hay error.
        """
        if not hasattr(self, "action"):
            self.build_slr_table()

        tokens = list(tokens) + [ENDMARK]
        stack = [0]  # pila de estados
        pos = 0

        while True:
            state = stack[-1]
            lookahead = tokens[pos]

            act = self.action.get(state, {}).get(lookahead)
            if act is None:
                print(f"Error de sintaxis en token {lookahead}, estado {state}")
                return False

            kind, value = act

            if kind == "shift":
                stack.append(value)
                pos += 1

            elif kind == "reduce":
                A, rhs = value
                # sacar |rhs| estados (si no es epsilon)
                if not (len(rhs) == 1 and rhs[0] == EPSILON):
                    for _ in range(len(rhs)):
                        stack.pop()
                t = stack[-1]
                goto_state = self.goto_table[t].get(A)
                if goto_state is None:
                    print(f"Error: no hay GOTO desde estado {t} con {A}")
                    return False
                stack.append(goto_state)
                print(f"reduce {A} -> {' '.join(rhs)}")

            elif kind == "accept":
                print("Cadena aceptada")
                return True


def export_lr0_to_dot(G: Grammar, filename: str = "lr0_automaton.dot"):
    """
    Genera un archivo .dot con el autómata LR(0) de la gramática.
    Cada estado se llama I0, I1, I2, ...
    """
    # Asegurarnos de que la colección LR(0) ya existe
    if not hasattr(G, "lr0_states"):
        G.build_lr0_collection()

    lines = []
    lines.append("digraph LR0 {")
    lines.append("  rankdir=LR;")
    lines.append('  node [shape=ellipse, fontname="Courier"];')

    # nodos: I0, I1, ...
    for idx, I in enumerate(G.lr0_states):
        items_lines = []

        # ordenar items por:
        # 1) símbolo de la izquierda según orden secuencial (G.symbol_rank)
        # 2) orden de la producción (G.prod_order)
        # 3) posición del punto (dot)
        sorted_items = sorted(
            I,
            key=lambda item: (
                G.symbol_rank.get(item[0], 10**9),
                G.prod_order.get((item[0], item[1]), 10**9),
                item[2],
            )
        )

        for (A, rhs, dot) in sorted_items:
            # construir producción con el punto
            rhs_with_dot = list(rhs)
            rhs_with_dot.insert(dot, "·")
            prod_str = f"{A} -> " + "".join(rhs_with_dot)
            items_lines.append(prod_str)

        # Agregar el número del estado (I0, I1, ...) como primera línea del label
        items_lines.insert(0, f"I{idx}")

        # unir cada item en una línea; escapar comillas
        label = "\\n".join(items_lines).replace('"', '\\"')
        lines.append(f'  I{idx} [label="{label}"];')

    # aristas: transiciones con símbolo X
    for (i, X), j in G.lr0_transitions.items():
        lines.append(f'  I{i} -> I{j} [label="{X}"];')

    lines.append("}")
    dot_source = "\n".join(lines)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(dot_source)

    print(f"Archivo DOT generado: {filename}")


# -------------------------
# Impresión de tablas SLR(1) con rich
# -------------------------
def print_slr_tables(G: Grammar):
    """
    Imprime las tablas ACTION y GOTO en formato tabular usando rich.
    """
    if not hasattr(G, "action"):
        G.build_slr_table()

    console = Console()

    terminals = sorted(list(G.terminals | {ENDMARK}))
    nonterminals = sorted(A for A in G.nonterminals if A != G.aug_start)
    num_states = len(G.lr0_states)

    # ACTION
    table_action = Table(title="ACTION TABLE")
    table_action.add_column("State", justify="right", style="bold")
    for a in terminals:
        table_action.add_column(a, justify="center")

    for s in range(num_states):
        row = [str(s)]
        for a in terminals:
            act = G.action.get(s, {}).get(a)
            if act is None:
                cell = ""
            else:
                kind, val = act
                if kind == "shift":
                    cell = f"s{val}"
                elif kind == "reduce":
                    A, rhs = val
                    cell = f"r {A}->{''.join(rhs)}"
                elif kind == "accept":
                    cell = "acc"
                else:
                    cell = "?"
            row.append(cell)
        table_action.add_row(*row)

    # GOTO
    table_goto = Table(title="GOTO TABLE")
    table_goto.add_column("State", justify="right", style="bold")
    for A in nonterminals:
        table_goto.add_column(A, justify="center")

    for s in range(num_states):
        row = [str(s)]
        for A in nonterminals:
            g = G.goto_table.get(s, {}).get(A)
            cell = "" if g is None else str(g)
            row.append(cell)
        table_goto.add_row(*row)

    console.print(table_action)
    console.print()
    console.print(table_goto)


# ==========================
# Ejemplo con la gramática:
#   S -> E
#   E -> E + T | T
#   T -> id
# ==========================
if __name__ == "__main__":
    grammar_prods = {
        'S': [['E']],
        'E': [['E', '+', 'T'], ['T']],
        'T': [['id']]
    }
    terminals = {'+', 'id'}

    G = Grammar(grammar_prods, start_symbol='S', terminals=terminals)
    console = Console()

    console.print("[bold]FIRST:[/bold]")
    for nt, s in G.compute_first_sets().items():
        if nt in G.nonterminals:
            console.print(f"{nt} : {s}")

    console.print("\n[bold]FOLLOW:[/bold]")
    for nt, s in G.compute_follow_sets().items():
        if nt in G.nonterminals:
            console.print(f"{nt} : {s}")

    console.print("\n[bold]Construyendo tabla SLR(1)...[/bold]")
    G.build_slr_table()
    print_slr_tables(G)

    # Exportar autómata LR(0) a DOT
    export_lr0_to_dot(G, filename="lr0_automaton.dot")

    # Estos tests no corresponden a esta gramática, puedes ajustarlos si quieres probarla de verdad
    tests = [
        ['b', 'b'],
        ['a', 'b', 'b'],
        ['a', 'a', 'b', 'b'],
        ['b'],
        ['a', 'b'],
    ]

    for w in tests:
        console.print(f"\n[bold]Probando:[/bold] {''.join(w)}")
        G.parse(w)
