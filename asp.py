import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from collections import defaultdict
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import warnings
import sys
from datetime import datetime
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
import copy
import os
warnings.filterwarnings('ignore')

# Устанавливаем кодировку для вывода
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# Определяем директорию скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Рабочая директория: {script_dir}")


class MultiRunSimulation:
    """
    Класс для управления множественными прогонами симуляции
    """
    
    def __init__(self, N_max, C, lambda_, beta, delta, n_runs=200, lambda_plus=None, lambda_minus=None):
        """
        Инициализация множественных прогонов
        n_runs: количество прогонов симуляции
        """
        self.N_max = N_max
        self.C = C
        self.lambda_ = lambda_
        self.beta = beta
        self.delta = delta
        self.lambda_plus = lambda_plus if lambda_plus is not None else lambda_
        self.lambda_minus = lambda_minus if lambda_minus is not None else lambda_
        self.n_runs = n_runs
        
        # Хранилище для всех прогонов
        self.runs = []
        self.all_histories = []
        self.all_states_histories = []
        
        # Агрегированные статистики
        self.aggregated = {
            'DS': [], 'DI': [], 'ES': [], 'EI': [], 'active': [], 'infected': []
        }
        self.aggregated_std = {
            'DS': [], 'DI': [], 'ES': [], 'EI': [], 'active': [], 'infected': []
        }
        
        # Для хранения текущего выбранного прогона
        self.current_run = 0
        
    def get_params_string(self):
        """
        Возвращает строку с параметрами для сохранения файлов
        """
        T_val = len(self.all_histories[0]['EI']) if self.all_histories else 0
        return (f"beta{self.beta:.2f}_delta{self.delta:.2f}_"
                f"lambda{self.lambda_:.2f}_C{self.C:.1f}_"
                f"T{T_val}")
    
    def run_all(self, T, verbose=True):
        """
        Запуск всех прогонов симуляции
        """
        self.runs = []
        self.all_histories = []
        self.all_states_histories = []
        
        for run_idx in range(self.n_runs):
            if verbose:
                print(f"Запуск прогона {run_idx + 1}/{self.n_runs}", end="\r")
            
            # Создаем и запускаем модель
            model = DynamicNetworkSIS(
                self.N_max, self.C, self.lambda_, self.beta, self.delta,
                self.lambda_plus, self.lambda_minus
            )
            model.initialize_random(p_enabled=0.7, p_infected=0.1)
            model.run(T, verbose=False)
            
            self.runs.append(model)
            self.all_histories.append(copy.deepcopy(model.history))
            self.all_states_histories.append(copy.deepcopy(model.states_history))
        
        # Агрегируем результаты
        self._aggregate_results()
        
        if verbose:
            print(f"\nВсе {self.n_runs} прогонов завершены")
    
    def _aggregate_results(self):
        """
        Агрегация результатов по всем прогонам
        """
        if not self.all_histories:
            return
        
        T = len(self.all_histories[0]['DS'])
        
        for key in self.aggregated.keys():
            # Собираем данные для каждого момента времени
            for t in range(T):
                values = [history[key][t] for history in self.all_histories]
                self.aggregated[key].append(np.mean(values))
                self.aggregated_std[key].append(np.std(values))
    
    def get_state_colors(self, step=None, run=None):
        """
        Возвращает цвета для визуализации графа на определенном шаге и прогоне
        """
        if run is None:
            run = self.current_run
            
        if run < len(self.all_states_histories):
            if step is not None and step < len(self.all_states_histories[run]):
                states = self.all_states_histories[run][step]
            else:
                states = self.all_states_histories[run][-1]
        else:
            states = np.zeros(self.N_max, dtype=int)
            
        color_map = []
        for state in states:
            if state == 0:
                color_map.append('lightgray')
            elif state == 1:
                color_map.append('darkgray')
            elif state == 2:
                color_map.append('lightblue')
            else:
                color_map.append('red')
        return color_map
    
    def get_graph(self, run=None):
        """Возвращает граф для указанного прогона"""
        if run is None:
            run = self.current_run
        return self.runs[run].graph
    
    def get_stats_text(self, step, run=None):
        """Возвращает текст статистики для определенного шага и прогона"""
        if run is None:
            run = self.current_run
            
        if run >= len(self.all_states_histories) or step >= len(self.all_states_histories[run]):
            return "Данные недоступны"
        
        states = self.all_states_histories[run][step]
        model = self.runs[run]
        
        active = np.where((states == 2) | (states == 3))[0]
        infected_active = np.where(states == 3)[0]
        infected_inactive = np.where(states == 1)[0]
        
        gamma = len(infected_active) / len(active) if len(active) > 0 else 0
        
        # Временно устанавливаем состояния в модели для расчета метрик
        original_states = model.states
        model.states = states
        metrics = model.calculate_advanced_metrics()
        model.states = original_states
        
        text = f"ПРОГОН {run+1}/{self.n_runs} | ШАГ {step}\n"
        text += "="*30 + "\n"
        text += f"DS: {np.sum(states == 0):3d} (неактивен, здоров)\n"
        text += f"DI: {np.sum(states == 1):3d} (неактивен, заражен)\n"
        text += f"ES: {np.sum(states == 2):3d} (активен, здоров)\n"
        text += f"EI: {np.sum(states == 3):3d} (активен, заражен)\n"
        text += "-"*30 + "\n"
        text += f"Активных: {len(active):3d}\n"
        text += f"Неактивных: {self.N_max - len(active):3d}\n"
        text += f"Зараженных всего: {len(infected_active) + len(infected_inactive):3d}\n"
        text += f"gamma: {gamma:.3f}\n"
        text += f"R0: {metrics.get('R0', 0):.2f}"
        
        return text
    
    def get_aggregated_stats_text(self, step):
        """Возвращает текст агрегированной статистики"""
        if step >= len(self.aggregated['EI']):
            return "Данные недоступны"
        
        text = f"АГРЕГИРОВАННЫЕ ДАННЫЕ (n={self.n_runs})\n"
        text += "="*30 + "\n"
        text += f"Параметры: beta={self.beta:.2f}, delta={self.delta:.2f}, lambda={self.lambda_:.2f}, C={self.C:.1f}\n"
        text += "-"*30 + "\n"
        
        metrics = ['DS', 'DI', 'ES', 'EI', 'active', 'infected']
        names = ['DS', 'DI', 'ES', 'EI', 'Активные', 'Зараженные']
        
        for metric, name in zip(metrics, names):
            mean = self.aggregated[metric][step]
            std = self.aggregated_std[metric][step]
            text += f"{name:10}: {mean:5.1f} +/- {std:4.1f}\n"
        
        # Добавляем информацию о разбросе
        text += "-"*30 + "\n"
        ei_values = [h['EI'][step] for h in self.all_histories]
        text += f"EI мин/макс: {np.min(ei_values):3.0f}/{np.max(ei_values):3.0f}\n"
        text += f"EI 25/75%: {np.percentile(ei_values, 25):3.0f}/{np.percentile(ei_values, 75):3.0f}"
        
        return text


class DynamicNetworkSIS:
    """
    Симуляция эпидемии на графе с динамической активностью узлов
    Состояния: 
    0 - (D,S) - неактивен и не заражен
    1 - (D,I) - неактивен и заражен
    2 - (E,S) - активен и не заражен
    3 - (E,I) - активен и заражен
    """
    
    def __init__(self, N_max, C, lambda_, beta, delta, lambda_plus=None, lambda_minus=None):
        self.N_max = N_max
        self.C = C
        self.beta = beta
        self.delta = delta
        
        self.lambda_plus = lambda_plus if lambda_plus is not None else lambda_
        self.lambda_minus = lambda_minus if lambda_minus is not None else lambda_
        
        # Проверка корректности параметров
        if self.lambda_plus < 0 or self.lambda_plus > 1:
            raise ValueError("lambda_plus должен быть в [0,1]")
        if self.lambda_minus < 0 or self.lambda_minus > 1:
            raise ValueError("lambda_minus должен быть в [0,1]")
        if self.beta < 0 or self.beta > 1:
            raise ValueError("beta должен быть в [0,1]")
        if self.delta < 0 or self.delta > 1:
            raise ValueError("delta должен быть в [0,1]")
        
        self.create_graph()
        
        # Хранение истории
        self.states_history = []
        self.history = defaultdict(list)
        self.current_step = 0
        self.states = np.zeros(N_max, dtype=int)
        
    def get_params_string(self):
        """
        Возвращает строку с параметрами для сохранения файлов
        """
        T_val = len(self.history['EI']) if self.history else 0
        return (f"beta{self.beta:.2f}_delta{self.delta:.2f}_"
                f"lambda{self.lambda_:.2f}_C{self.C:.1f}_"
                f"T{T_val}")
        
    def create_graph(self):
        """Создание случайного графа"""
        if self.N_max <= 1:
            self.graph = nx.Graph()
            self.graph.add_node(0)
            return
            
        p = min(1.0, self.C / (self.N_max - 1))
        self.graph = nx.erdos_renyi_graph(self.N_max, p, seed=None)
        
        # Убеждаемся, что граф связный
        if self.N_max > 1 and not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                self.graph.add_edge(node1, node2)
    
    def initialize_random(self, p_enabled=0.7, p_infected=0.1):
        """Случайная инициализация состояний"""
        for i in range(self.N_max):
            enabled = np.random.random() < p_enabled
            infected = np.random.random() < p_infected if enabled else False
            
            if not enabled and not infected:
                self.states[i] = 0  # DS
            elif not enabled and infected:
                self.states[i] = 1  # DI
            elif enabled and not infected:
                self.states[i] = 2  # ES
            else:
                self.states[i] = 3  # EI
                
    def get_active_nodes(self, states=None):
        """Возвращает индексы активных узлов (E)"""
        if states is None:
            states = self.states
        return np.where((states == 2) | (states == 3))[0]
    
    def get_infected_active(self, states=None):
        """Возвращает индексы зараженных активных узлов (EI)"""
        if states is None:
            states = self.states
        return np.where(states == 3)[0]
    
    def get_infected_inactive(self, states=None):
        """Возвращает индексы зараженных неактивных узлов (DI)"""
        if states is None:
            states = self.states
        return np.where(states == 1)[0]
    
    def compute_mu(self, node, states=None):
        """
        Вычисляет вероятность заражения для узла
        mu = beta * c * gamma, где gamma = I_active / N_active
        """
        if states is None:
            states = self.states
            
        active_nodes = self.get_active_nodes(states)
        infected_active = self.get_infected_active(states)
        
        N_active = len(active_nodes)
        I_active = len(infected_active)
        
        if N_active == 0:
            return 0
        
        gamma = I_active / N_active
        c = self.graph.degree(node)
        
        mu = self.beta * c * gamma
        return min(mu, 1.0)
    
    def calculate_advanced_metrics(self, step=None):
        """Расширенная аналитика"""
        if step is not None and step < len(self.states_history):
            states = self.states_history[step]
        else:
            states = self.states
            
        metrics = {}
        
        # Метрики центральности зараженных узлов
        infected_nodes = np.where((states == 1) | (states == 3))[0]
        if len(infected_nodes) > 0:
            metrics['avg_degree_infected'] = np.mean([self.graph.degree(n) for n in infected_nodes])
            try:
                metrics['clustering_infected'] = nx.average_clustering(self.graph, nodes=infected_nodes)
            except:
                metrics['clustering_infected'] = 0
        
        # Скорость распространения
        if len(self.history['EI']) > 1:
            ei_diff = np.diff(self.history['EI'])
            metrics['avg_spread_rate'] = np.mean(ei_diff[ei_diff > 0]) if any(ei_diff > 0) else 0
            metrics['max_spread_rate'] = np.max(ei_diff) if len(ei_diff) > 0 else 0
        
        # Время до пика инфекции
        if len(self.history['EI']) > 0:
            metrics['peak_infection'] = np.max(self.history['EI'])
            metrics['time_to_peak'] = np.argmax(self.history['EI'])
        
        # Энтропия распределения состояний
        state_counts = [np.sum(states == i) for i in range(4)]
        probs = [c/self.N_max for c in state_counts]
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
        metrics['state_entropy'] = entropy
        
        # Репродуктивное число R0
        avg_degree_all = np.mean([self.graph.degree(i) for i in range(self.N_max)])
        metrics['R0'] = self.beta * avg_degree_all / self.delta if self.delta > 0 else float('inf')
        
        return metrics
    
    def step(self):
        """Один шаг симуляции"""
        new_states = self.states.copy()
        
        active_nodes = self.get_active_nodes()
        infected_active = self.get_infected_active()
        N_active = len(active_nodes)
        I_active = len(infected_active)
        gamma = I_active / N_active if N_active > 0 else 0
        
        for i in range(self.N_max):
            state = self.states[i]
            r = np.random.random()
            
            if state == 0:  # DS
                if r < self.lambda_plus:
                    new_states[i] = 2  # ES
                    
            elif state == 1:  # DI
                if r < self.lambda_plus:
                    new_states[i] = 3  # EI
                    
            elif state == 2:  # ES
                mu = self.beta * self.graph.degree(i) * gamma
                mu = min(mu, 1.0)
                if r < self.lambda_minus:
                    new_states[i] = 0  # DS
                elif r < self.lambda_minus + mu:
                    new_states[i] = 3  # EI
                    
            elif state == 3:  # EI
                if r < self.lambda_minus:
                    new_states[i] = 1  # DI
                elif r < self.lambda_minus + self.delta:
                    new_states[i] = 2  # ES
        
        self.states = new_states
        self.current_step += 1
        self.record_history()
    
    def record_history(self):
        """Запись текущего состояния для истории"""
        counts = {
            'DS': np.sum(self.states == 0),
            'DI': np.sum(self.states == 1),
            'ES': np.sum(self.states == 2),
            'EI': np.sum(self.states == 3),
            'active': np.sum((self.states == 2) | (self.states == 3)),
            'infected': np.sum((self.states == 1) | (self.states == 3))
        }
        for key, value in counts.items():
            self.history[key].append(value)
            
    def run(self, T, verbose=True):
        """Запуск симуляции на T шагов с сохранением всей истории"""
        self.states_history = []
        self.history.clear()
        
        # Сохраняем начальное состояние
        self.states_history.append(self.states.copy())
        self.record_history()
        self.current_step = 0
        
        for step in range(T):
            self.step()
            self.states_history.append(self.states.copy())
            if verbose and (step + 1) % 10 == 0:
                print(f"Шаг {step + 1}/{T} завершен")
    
    def get_statistics(self, step=None):
        """Возвращает статистику"""
        if step is not None and step < len(self.states_history):
            states = self.states_history[step]
        else:
            states = self.states
            
        active = self.get_active_nodes(states)
        infected_active = self.get_infected_active(states)
        infected_inactive = self.get_infected_inactive(states)
        
        stats = {
            'total_nodes': self.N_max,
            'active_nodes': len(active),
            'inactive_nodes': self.N_max - len(active),
            'infected_total': len(infected_active) + len(infected_inactive),
            'infected_active': len(infected_active),
            'infected_inactive': len(infected_inactive),
            'healthy_active': np.sum(states == 2),
            'healthy_inactive': np.sum(states == 0),
            'gamma': len(infected_active) / len(active) if len(active) > 0 else 0,
            'avg_degree': np.mean([self.graph.degree(i) for i in range(self.N_max)])
        }
        return stats
    
    def save_plot(self, fig, plot_name):
        """
        Сохраняет график с именем, содержащим параметры модели
        """
        filename = f"{plot_name}_{self.get_params_string()}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {filename}")
        return filename
    
    def plot_risk_heatmap(self):
        """Тепловая карта риска заражения для каждого узла"""
        if len(self.states_history) == 0:
            print("Нет данных для визуализации")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Добавляем общий заголовок с параметрами
        fig.suptitle(f'Тепловая карта риска: beta={self.beta:.2f}, delta={self.delta:.2f}, lambda={self.lambda_:.2f}, C={self.C:.1f}', 
                    fontsize=14, fontweight='bold')
        
        # Тепловая карта риска по времени
        risk_over_time = []
        for t in range(len(self.states_history)):
            time_risks = [self.compute_mu(n, self.states_history[t]) for n in range(self.N_max)]
            risk_over_time.append(time_risks)
        
        risk_matrix = np.array(risk_over_time).T
        
        im = ax1.imshow(risk_matrix, aspect='auto', cmap='hot', 
                       interpolation='nearest', 
                       extent=[0, len(self.states_history)-1, 0, self.N_max])
        ax1.set_xlabel('Время')
        ax1.set_ylabel('Узел')
        ax1.set_title('Тепловая карта риска заражения')
        plt.colorbar(im, ax=ax1, label='Вероятность заражения mu')
        
        # Средний риск по времени для каждого узла
        avg_risks = np.mean(risk_matrix, axis=1)
        sorted_idx = np.argsort(avg_risks)[::-1]
        
        ax2.barh(range(min(20, self.N_max)), 
                [avg_risks[sorted_idx[i]] for i in range(min(20, self.N_max))])
        ax2.set_yticks(range(min(20, self.N_max)))
        ax2.set_yticklabels([f'Узел {sorted_idx[i]}' for i in range(min(20, self.N_max))])
        ax2.set_xlabel('Средний риск')
        ax2.set_title(f'Топ-{min(20, self.N_max)} узлов с наибольшим риском')
        
        # Зависимость риска от степени узла
        degrees = [self.graph.degree(i) for i in range(self.N_max)]
        ax3.scatter(degrees, avg_risks, alpha=0.6, c='red', edgecolors='black')
        ax3.set_xlabel('Степень узла')
        ax3.set_ylabel('Средний риск')
        ax3.set_title('Зависимость риска от степени узла')
        
        # Линия тренда
        if len(degrees) > 1:
            z = np.polyfit(degrees, avg_risks, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(degrees), max(degrees), 100)
            ax3.plot(x_range, p(x_range), "b--", alpha=0.8, 
                    label=f'Тренд (R²={np.corrcoef(degrees, avg_risks)[0,1]**2:.3f})')
            ax3.legend()
        
        plt.tight_layout()
        
        # Сохраняем график
        self.save_plot(fig, "risk_heatmap")
        
        plt.show()


class Interactive3DGraph:
    """
    Класс для интерактивного 3D графа с возможностью клика по узлам и зумом
    """
    def __init__(self, ax, graph, pos_3d, colors, sizes, node_labels=None, params_string=""):
        self.ax = ax
        self.graph = graph
        self.pos_3d = pos_3d
        self.colors = colors
        self.sizes = sizes
        self.node_labels = node_labels if node_labels else {}
        self.params_string = params_string
        
        # Для подсветки выбранного узла
        self.selected_node = None
        
        # Сохраняем начальные пределы для сброса зума
        x_vals = [self.pos_3d[node][0] for node in self.graph.nodes]
        y_vals = [self.pos_3d[node][1] for node in self.graph.nodes]
        z_vals = [self.pos_3d[node][2] for node in self.graph.nodes]
        margin = 0.5
        self.initial_limits = {
            'xlim': (min(x_vals) - margin, max(x_vals) + margin),
            'ylim': (min(y_vals) - margin, max(y_vals) + margin),
            'zlim': (min(z_vals) - margin, max(z_vals) + margin)
        }
        
        # Для отслеживания текущих пределов
        self.current_limits = self.initial_limits.copy()
        
        # Рисуем начальное состояние
        self.draw_graph()
        
        # Подключаем обработчики
        self.cid_pick = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid_scroll = self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        
    def draw_graph(self):
        """Отрисовка графа с возможностью подсветки узла"""
        self.ax.clear()
        
        # Добавляем заголовок с параметрами
        if self.params_string:
            formatted_params = self.params_string.replace('_', ', ').replace('beta', 'beta=').replace('delta', 'delta=').replace('lambda', 'lambda=')
            self.ax.set_title(f'3D Сеть: {formatted_params}\n(кликните на узел для подсветки связей)', fontsize=10)
        
        # Рисуем все ребра
        for edge in self.graph.edges():
            x = [self.pos_3d[edge[0]][0], self.pos_3d[edge[1]][0]]
            y = [self.pos_3d[edge[0]][1], self.pos_3d[edge[1]][1]]
            z = [self.pos_3d[edge[0]][2], self.pos_3d[edge[1]][2]]
            
            # Если ребро связано с выделенным узлом, рисуем ярче и толще
            if self.selected_node is not None and (edge[0] == self.selected_node or edge[1] == self.selected_node):
                self.ax.plot(x, y, z, color='yellow', alpha=0.9, linewidth=3)
            else:
                self.ax.plot(x, y, z, color='gray', alpha=0.3, linewidth=0.8)
        
        # Рисуем узлы
        for node in range(len(self.graph.nodes)):
            # Размер узла
            size = self.sizes[node] * 2 if node == self.selected_node else self.sizes[node]
            
            # Цвет ободка
            edgecolor = 'yellow' if node == self.selected_node else 'black'
            linewidth = 3 if node == self.selected_node else 0.5
            
            scatter = self.ax.scatter([self.pos_3d[node][0]], 
                                     [self.pos_3d[node][1]], 
                                     [self.pos_3d[node][2]],
                                     c=[self.colors[node]], 
                                     s=[size], 
                                     alpha=1.0 if node == self.selected_node else 0.8,
                                     edgecolors=edgecolor, 
                                     linewidth=linewidth,
                                     picker=True)
        
        # Добавляем подписи для узлов (если их не слишком много)
        if len(self.graph.nodes) <= 30:
            for node in self.graph.nodes:
                label = self.node_labels.get(node, str(node))
                self.ax.text(self.pos_3d[node][0], self.pos_3d[node][1], self.pos_3d[node][2], 
                           label, fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Восстанавливаем пределы (сохраняем текущий зум)
        self.ax.set_xlim(self.current_limits['xlim'])
        self.ax.set_ylim(self.current_limits['ylim'])
        self.ax.set_zlim(self.current_limits['zlim'])
        
        self.ax.figure.canvas.draw_idle()
    
    def on_pick(self, event):
        """Обработчик выбора узла"""
        # Находим, какой узел был выбран
        if hasattr(event, 'ind') and len(event.ind) > 0:
            # Получаем координаты выбранной точки
            artist = event.artist
            if hasattr(artist, '_offsets3d'):
                offsets = artist._offsets3d
                if offsets and len(offsets) == 3:
                    idx = event.ind[0]
                    x, y, z = offsets[0][idx], offsets[1][idx], offsets[2][idx]
                    
                    # Ищем ближайший узел
                    min_dist = float('inf')
                    closest_node = None
                    for node in self.graph.nodes:
                        node_pos = self.pos_3d[node]
                        dist = np.sqrt((x - node_pos[0])**2 + 
                                     (y - node_pos[1])**2 + 
                                     (z - node_pos[2])**2)
                        if dist < min_dist and dist < 0.5:
                            min_dist = dist
                            closest_node = node
                    
                    if closest_node is not None:
                        if self.selected_node == closest_node:
                            # Если кликнули на тот же узел, снимаем выделение
                            self.selected_node = None
                            print(f"Выделение снято")
                        else:
                            # Выделяем новый узел
                            self.selected_node = closest_node
                            print(f"Выделен узел {closest_node}, степень: {self.graph.degree(closest_node)}")
                        
                        # Перерисовываем с сохранением текущего зума
                        self.draw_graph()
    
    def on_scroll(self, event):
        """Обработчик прокрутки колесика мыши для зума"""
        if event.inaxes != self.ax:
            return
        
        # Коэффициент масштабирования
        scale_factor = 1.1 if event.button == 'up' else 0.9
        
        # Получаем текущие пределы
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
        # Вычисляем центр текущего вида
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # Новые пределы с масштабированием относительно центра
        x_new = (x_center + (xlim[0] - x_center) * scale_factor,
                 x_center + (xlim[1] - x_center) * scale_factor)
        y_new = (y_center + (ylim[0] - y_center) * scale_factor,
                 y_center + (ylim[1] - y_center) * scale_factor)
        z_new = (z_center + (zlim[0] - z_center) * scale_factor,
                 z_center + (zlim[1] - z_center) * scale_factor)
        
        # Устанавливаем новые пределы
        self.ax.set_xlim(x_new)
        self.ax.set_ylim(y_new)
        self.ax.set_zlim(z_new)
        
        # Сохраняем текущие пределы
        self.current_limits = {
            'xlim': x_new,
            'ylim': y_new,
            'zlim': z_new
        }
        
        self.ax.figure.canvas.draw_idle()
    
    def reset_view(self):
        """Сброс вида к начальным пределам"""
        self.current_limits = self.initial_limits.copy()
        self.draw_graph()
    
    def update_colors_and_sizes(self, colors, sizes):
        """Обновление цветов и размеров узлов"""
        self.colors = colors
        self.sizes = sizes
        self.draw_graph()


def save_current_view(fig, multi_sim, plot_name="unified_view"):
    """
    Сохраняет текущий вид с параметрами в имени файла
    """
    filename = f"{plot_name}_{multi_sim.get_params_string()}.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Текущий вид сохранен: {filename}")
    return filename


def create_unified_viewer(multi_sim):
    """
    Создает унифицированный интерактивный просмотр для множественных прогонов с 3D графом
    и возможностью клика по узлам
    """
    if multi_sim.n_runs == 0 or len(multi_sim.all_histories) == 0:
        print("Нет данных для визуализации. Сначала запустите симуляцию.")
        return
    
    fig = plt.figure(figsize=(22, 12))
    
    # Добавляем общий заголовок с параметрами
    params_text = (f"beta={multi_sim.beta:.2f}, delta={multi_sim.delta:.2f}, "
                   f"lambda={multi_sim.lambda_:.2f}, C={multi_sim.C:.1f}, "
                   f"T={len(multi_sim.all_histories[0]['EI'])}")
    fig.suptitle(f'УНИФИЦИРОВАННЫЙ ИНТЕРАКТИВНЫЙ ПРОСМОТР: {params_text}', 
                fontsize=14, fontweight='bold')
    
    # Создаем сетку для размещения графиков
    gs = fig.add_gridspec(3, 5, height_ratios=[2, 1, 0.2], hspace=0.3, wspace=0.3)
    
    # 3D граф будет в первой строке, первые две колонки
    ax_graph_3d = fig.add_subplot(gs[0, :2], projection='3d')
    ax_dynamics = fig.add_subplot(gs[0, 2:])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_distribution = fig.add_subplot(gs[1, 2])
    ax_stats = fig.add_subplot(gs[1, 3:])
    
    # Бегунок и кнопки управления
    ax_controls = fig.add_subplot(gs[2, :])
    ax_controls.axis('off')
    
    # Получаем граф для первого прогона
    graph = multi_sim.get_graph(0)
    # Вычисляем 3D позиции для узлов (фиксируем, чтобы не менялись)
    pos_3d = nx.spring_layout(graph, dim=3, seed=42, k=2, iterations=50)
    
    # Создаем слайдеры
    ax_time_slider = plt.axes([0.15, 0.02, 0.25, 0.03])
    time_slider = Slider(ax_time_slider, 'Время', 0, 
                        len(multi_sim.all_histories[0]['EI'])-1, 
                        valinit=len(multi_sim.all_histories[0]['EI'])-1, 
                        valstep=1)
    
    ax_run_slider = plt.axes([0.55, 0.02, 0.15, 0.03])
    run_slider = Slider(ax_run_slider, 'Прогон', 1, multi_sim.n_runs, 
                       valinit=1, valstep=1)
    
    # Кнопки
    ax_play = plt.axes([0.75, 0.015, 0.05, 0.04])
    play_button = Button(ax_play, '▶')
    
    ax_pause = plt.axes([0.81, 0.015, 0.05, 0.04])
    pause_button = Button(ax_pause, '⏸')
    
    # Кнопка сохранения
    ax_save = plt.axes([0.88, 0.15, 0.1, 0.03])
    save_button = Button(ax_save, 'Сохранить')
    
    # Кнопка сброса выделения
    ax_clear = plt.axes([0.88, 0.06, 0.1, 0.03])
    clear_button = Button(ax_clear, 'Снять выделение')
    
    # Кнопка сброса зума
    ax_reset_view = plt.axes([0.88, 0.1, 0.1, 0.03])
    reset_view_button = Button(ax_reset_view, 'Сбросить вид')
    
    # Чекбокс для отображения среднего
    ax_check = plt.axes([0.88, 0.015, 0.1, 0.04])
    check = CheckButtons(ax_check, ['Показать среднее'], [True])
    
    # Переменные для анимации и интерактивности
    anim_running = False
    interactive_graph = None
    
    def update_plots(val):
        nonlocal interactive_graph
        time_step = int(time_slider.val)
        run_idx = int(run_slider.val) - 1
        show_avg = check.get_status()[0]
        
        # Очищаем оси (кроме 3D графа, он будет обновлен через Interactive3DGraph)
        ax_dynamics.clear()
        ax_hist.clear()
        ax_metrics.clear()
        ax_distribution.clear()
        ax_stats.clear()
        
        # 1. 3D Граф сети через интерактивный класс
        colors = multi_sim.get_state_colors(time_step, run_idx)
        
        # Размер узлов зависит от степени (нормируем для наглядности)
        degrees = [graph.degree(node) for node in range(multi_sim.N_max)]
        max_deg = max(degrees) if degrees else 1
        sizes = [50 + 30 * (d / max_deg) for d in degrees]
        
        # Создаем или обновляем интерактивный граф
        if interactive_graph is None:
            # Для первого раза создаем
            node_labels = {i: str(i) for i in range(multi_sim.N_max)} if multi_sim.N_max <= 30 else {}
            interactive_graph = Interactive3DGraph(ax_graph_3d, graph, pos_3d, colors, sizes, node_labels, multi_sim.get_params_string())
        else:
            # Обновляем цвета и размеры, сохраняя выделение
            interactive_graph.update_colors_and_sizes(colors, sizes)
        
        # 2. График динамики
        time = range(len(multi_sim.all_histories[0]['EI']))
        
        # Рисуем все траектории полупрозрачными
        if show_avg:
            for i in range(min(20, multi_sim.n_runs)):
                ax_dynamics.plot(time, multi_sim.all_histories[i]['EI'], 
                               color='gray', alpha=0.1, linewidth=0.5)
            
            # Средняя траектория
            ax_dynamics.plot(time, multi_sim.aggregated['EI'], 
                           color='red', linewidth=3, label='Среднее EI')
            ax_dynamics.fill_between(time,
                np.array(multi_sim.aggregated['EI']) - np.array(multi_sim.aggregated_std['EI']),
                np.array(multi_sim.aggregated['EI']) + np.array(multi_sim.aggregated_std['EI']),
                color='red', alpha=0.2, label='±1 std')
        
        # Текущий прогон
        ax_dynamics.plot(time, multi_sim.all_histories[run_idx]['EI'], 
                        color='blue', linewidth=2.5, label=f'Прогон {run_idx+1}')
        
        # Вертикальная линия текущего времени
        ax_dynamics.axvline(x=time_step, color='green', linestyle='--', linewidth=2)
        
        ax_dynamics.set_xlabel('Время')
        ax_dynamics.set_ylabel('Количество EI')
        ax_dynamics.set_title('Динамика зараженных активных узлов')
        ax_dynamics.legend(loc='upper right', fontsize=8)
        ax_dynamics.grid(True, alpha=0.3)
        
        # 3. Гистограмма распределения состояний с подписями значений
        if show_avg:
            # Показываем среднее распределение
            states_data = []
            for run in range(multi_sim.n_runs):
                states = multi_sim.all_states_histories[run][time_step]
                states_data.append(states)
            
            avg_counts = []
            for state in range(4):
                avg_counts.append(np.mean([np.sum(s == state) for s in states_data]))
            
            bars = ax_hist.bar(['DS', 'DI', 'ES', 'EI'], avg_counts,
                              color=['lightgray', 'darkgray', 'lightblue', 'red'],
                              alpha=0.5, edgecolor='black', label='Среднее')
            
            # Добавляем подписи значений над столбцами
            for bar, count in zip(bars, avg_counts):
                height = bar.get_height()
                ax_hist.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{count:.1f}', ha='center', va='bottom', 
                            fontweight='bold', fontsize=9, color='black')
            
            # Текущий прогон (кружки с подписями)
            current_states = multi_sim.all_states_histories[run_idx][time_step]
            current_counts = [np.sum(current_states == i) for i in range(4)]
            
            # Рисуем кружки
            scatter = ax_hist.scatter(['DS', 'DI', 'ES', 'EI'], current_counts,
                          color='black', s=150, zorder=5, label=f'Прогон {run_idx+1}')
            
            # Добавляем подписи к кружкам
            for i, (state, count) in enumerate(zip(['DS', 'DI', 'ES', 'EI'], current_counts)):
                ax_hist.annotate(f'{count}', 
                               xy=(i, count),
                               xytext=(0, 10),
                               textcoords='offset points',
                               ha='center', va='bottom',
                               fontweight='bold', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        else:
            # Только текущий прогон
            states = multi_sim.all_states_histories[run_idx][time_step]
            counts = [np.sum(states == i) for i in range(4)]
            bars = ax_hist.bar(['DS', 'DI', 'ES', 'EI'], counts,
                              color=['lightgray', 'darkgray', 'lightblue', 'red'],
                              edgecolor='black', alpha=0.7)
            
            # Добавляем значения над столбцами
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax_hist.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{count}', ha='center', va='bottom', 
                            fontweight='bold', fontsize=10)
        
        ax_hist.set_title('Распределение состояний')
        ax_hist.set_ylabel('Количество')
        ax_hist.grid(True, alpha=0.3, axis='y')
        if show_avg:
            ax_hist.legend(loc='upper right', fontsize=8)
        
        # 4. Ключевые метрики
        if show_avg:
            # Метрики для среднего
            # Правильно вычисляем среднюю степень по всем узлам
            first_run = multi_sim.runs[0]
            avg_degree = np.mean([first_run.graph.degree(i) for i in range(multi_sim.N_max)])
            r0_value = first_run.beta * avg_degree / first_run.delta if first_run.delta > 0 else float('inf')
            
            avg_metrics = {
                'R0': r0_value,
                'Пик EI': np.max(multi_sim.aggregated['EI']),
                'Текущий EI': multi_sim.aggregated['EI'][time_step],
                'Активных': multi_sim.aggregated['active'][time_step]
            }
            metrics_to_show = avg_metrics
        else:
            # Метрики для текущего прогона (здесь все правильно)
            model = multi_sim.runs[run_idx]
            model.states = multi_sim.all_states_histories[run_idx][time_step]
            metrics = model.calculate_advanced_metrics()
            metrics_to_show = {
                'R0': metrics.get('R0', 0),
                'Пик EI': metrics.get('peak_infection', 0),
                'Скорость': metrics.get('avg_spread_rate', 0),
                'Энтропия': metrics.get('state_entropy', 0)
            }
        
        y_pos = np.arange(len(metrics_to_show))
        ax_metrics.barh(y_pos, list(metrics_to_show.values()), color='steelblue')
        ax_metrics.set_yticks(y_pos)
        ax_metrics.set_yticklabels(list(metrics_to_show.keys()))
        ax_metrics.set_title('Ключевые метрики')
        ax_metrics.grid(True, alpha=0.3, axis='x')
        
        # Добавляем значения на горизонтальные столбцы
        for i, (key, value) in enumerate(metrics_to_show.items()):
            ax_metrics.text(value + 0.5, i, f'{value:.2f}', 
                          va='center', fontsize=9, fontweight='bold')
        
        # 5. Распределение прогонов в текущий момент
        ei_values = [h['EI'][time_step] for h in multi_sim.all_histories]
        n, bins, patches = ax_distribution.hist(ei_values, bins=min(20, multi_sim.n_runs//2), 
                           color='lightcoral', edgecolor='black', alpha=0.7)
        
        # Добавляем подписи количества над столбцами гистограммы
        for i, (count, bin_edge) in enumerate(zip(n, bins[:-1])):
            if count > 0:
                ax_distribution.text(bin_edge + (bins[1]-bins[0])/2, count + 0.5,
                                    f'{int(count)}', ha='center', va='bottom',
                                    fontsize=8)
        
        ax_distribution.axvline(x=multi_sim.all_histories[run_idx]['EI'][time_step],
                               color='blue', linewidth=3, label=f'Прогон {run_idx+1}')
        ax_distribution.axvline(x=multi_sim.aggregated['EI'][time_step],
                               color='red', linewidth=3, linestyle='--', label='Среднее')
        ax_distribution.set_xlabel('Количество EI')
        ax_distribution.set_ylabel('Частота')
        ax_distribution.set_title(f'Распределение EI в t={time_step}')
        ax_distribution.legend(loc='upper right', fontsize=8)
        ax_distribution.grid(True, alpha=0.3)
        
        # 6. Статистика
        ax_stats.axis('off')
        if show_avg:
            stats_text = multi_sim.get_aggregated_stats_text(time_step)
        else:
            stats_text = multi_sim.get_stats_text(time_step, run_idx)
        
        ax_stats.text(0.05, 0.95, stats_text, fontsize=10, 
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        fig.canvas.draw_idle()
    
    def animate():
        nonlocal anim_running
        anim_running = True
        current = time_slider.val
        while anim_running and current < time_slider.valmax:
            current += 1
            time_slider.set_val(current)
            plt.pause(0.2)
    
    def play(event):
        nonlocal anim_running
        if not anim_running:
            animate()
    
    def pause(event):
        nonlocal anim_running
        anim_running = False
    
    def save_current(event):
        save_current_view(fig, multi_sim, f"unified_view_run{int(run_slider.val)}_t{int(time_slider.val)}")
    
    def clear_selection(event):
        nonlocal interactive_graph
        if interactive_graph:
            interactive_graph.selected_node = None
            interactive_graph.draw_graph()
    
    def reset_view(event):
        nonlocal interactive_graph
        if interactive_graph:
            interactive_graph.reset_view()
    
    # Подключаем обработчики
    time_slider.on_changed(update_plots)
    run_slider.on_changed(update_plots)
    check.on_clicked(update_plots)
    play_button.on_clicked(play)
    pause_button.on_clicked(pause)
    save_button.on_clicked(save_current)
    clear_button.on_clicked(clear_selection)
    reset_view_button.on_clicked(reset_view)
    
    # Начальное обновление
    update_plots(None)
    
    plt.show()

def run_simulation_with_params(auto_mode=False):
    """Запуск симуляции с вводом всех параметров"""
    print("="*70)
    print("МОДЕЛИРОВАНИЕ ЭПИДЕМИИ НА ДИНАМИЧЕСКОМ ГРАФЕ".center(70))
    print("="*70)
    
    try:
        # Если автоматический режим - берем параметры из командной строки
        if auto_mode or len(sys.argv) > 1:
            if len(sys.argv) >= 8:
                N_max = int(sys.argv[1])
                C = float(sys.argv[2])
                lambda_ = float(sys.argv[3])
                beta = float(sys.argv[4])
                delta = float(sys.argv[5])
                T = int(sys.argv[6])
                n_runs = int(sys.argv[7])
                print(f"\nАвтоматический режим с параметрами:")
                print(f"N_max={N_max}, C={C}, lambda={lambda_}, beta={beta}, delta={delta}, T={T}, n_runs={n_runs}")
            else:
                print("Ошибка: недостаточно параметров в командной строке")
                return
        else:
            # Интерактивный режим
            print("\nВВЕДИТЕ ПАРАМЕТРЫ МОДЕЛИ:")
            N_max = int(input("  Максимальное количество узлов (N_max) [100]: ") or "100")
            C = float(input("  Средняя степень узла (C) [4.0]: ") or "4.0")
            lambda_ = float(input("  Вероятность смены активности (lambda) [0.2]: ") or "0.2")
            beta = float(input("  Вероятность передачи инфекции (beta) [0.3]: ") or "0.3")
            delta = float(input("  Вероятность выздоровления (delta) [0.2]: ") or "0.2")
            T = int(input("  Количество шагов симуляции (T) [100]: ") or "100")
            n_runs = int(input("  Количество прогонов [50]: ") or "50")
        
        # Проверка допустимости параметров
        if beta < 0 or beta > 1:
            print("Ошибка: beta должен быть в [0,1]")
            return
        if delta < 0 or delta > 1:
            print("Ошибка: delta должен быть в [0,1]")
            return
        if lambda_ < 0 or lambda_ > 1:
            print("Ошибка: lambda должен быть в [0,1]")
            return
        
        print("\n" + "="*70)
        print("ЗАПУСК СИМУЛЯЦИИ".center(70))
        print("="*70)
        print(f"Параметры: N_max={N_max}, C={C}, lambda={lambda_}, beta={beta}, delta={delta}, T={T}, прогонов={n_runs}")
        
        print(f"\nЗапуск {n_runs} прогонов симуляции...")
        multi_sim = MultiRunSimulation(N_max, C, lambda_, beta, delta, n_runs)
        multi_sim.run_all(T)
        
        # Сохраняем агрегированные данные в CSV
        agg_df = pd.DataFrame({
            'time': range(len(multi_sim.aggregated['EI'])),
            **{f"{key}_mean": multi_sim.aggregated[key] for key in multi_sim.aggregated.keys()},
            **{f"{key}_std": multi_sim.aggregated_std[key] for key in multi_sim.aggregated_std.keys()}
        })
        csv_filename = f"aggregated_{multi_sim.get_params_string()}.csv"
        agg_df.to_csv(csv_filename, index=False)
        print(f"Агрегированные данные сохранены: {csv_filename}")
        
        # В автоматическом режиме не показываем графики
        if not auto_mode and len(sys.argv) <= 1:
            # Спрашиваем, показывать ли интерактивный просмотр
            show_viewer = input("\nПоказать интерактивный просмотр? (y/n) [y]: ") or "y"
            if show_viewer.lower() == 'y':
                print("\nЗапуск интерактивного просмотра...")
                create_unified_viewer(multi_sim)
            
            # Дополнительное меню для анализа
            while True:
                print("\n" + "="*70)
                print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ".center(70))
                print("="*70)
                print("1. Показать все траектории для метрики")
                print("2. Показать доверительные полосы")
                print("3. Анализ сходимости")
                print("4. Тепловая карта риска")
                print("0. Выход")
                
                choice = input("\nВыберите опцию: ")
                
                if choice == "1":
                    metric = input("Метрика [EI]: ") or "EI"
                    if metric in ['DS', 'DI', 'ES', 'EI', 'active', 'infected']:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        plt.suptitle(f'Траектории {metric}: {multi_sim.get_params_string().replace("_", ", ")}')
                        time = range(len(multi_sim.aggregated[metric]))
                        
                        for i in range(multi_sim.n_runs):
                            ax.plot(time, multi_sim.all_histories[i][metric], color='gray', alpha=0.2, linewidth=0.8)
                        
                        ax.plot(time, multi_sim.aggregated[metric], color='red', linewidth=3, label='Среднее')
                        ax.fill_between(time,
                            np.array(multi_sim.aggregated[metric]) - np.array(multi_sim.aggregated_std[metric]),
                            np.array(multi_sim.aggregated[metric]) + np.array(multi_sim.aggregated_std[metric]),
                            color='red', alpha=0.2, label='±1 std')
                        
                        ax.set_xlabel('Время')
                        ax.set_ylabel(f'Количество {metric}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        filename = f"trajectories_{metric}_{multi_sim.get_params_string()}.png"
                        fig.savefig(filename, dpi=150, bbox_inches='tight')
                        print(f"График сохранен: {filename}")
                        plt.show()
                        
                elif choice == "2":
                    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
                    axes = axes.flatten()
                    metrics = ['DS', 'DI', 'ES', 'EI', 'active', 'infected']
                    colors = ['lightgray', 'darkgray', 'lightblue', 'red', 'blue', 'darkred']
                    titles = ['DS', 'DI', 'ES', 'EI', 'Активные', 'Зараженные']
                    
                    fig.suptitle(f'Доверительные полосы: {multi_sim.get_params_string().replace("_", ", ")}')
                    time = range(len(multi_sim.aggregated[metrics[0]]))
                    
                    for idx, (metric, color, title) in enumerate(zip(metrics, colors, titles)):
                        ax = axes[idx]
                        ax.plot(time, multi_sim.aggregated[metric], color=color, linewidth=2.5)
                        ax.fill_between(time,
                            np.array(multi_sim.aggregated[metric]) - np.array(multi_sim.aggregated_std[metric]),
                            np.array(multi_sim.aggregated[metric]) + np.array(multi_sim.aggregated_std[metric]),
                            color=color, alpha=0.2)
                        ax.set_xlabel('Время')
                        ax.set_ylabel('Количество')
                        ax.set_title(title)
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    filename = f"confidence_bands_{multi_sim.get_params_string()}.png"
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"График сохранен: {filename}")
                    plt.show()
                    
                elif choice == "3":
                    metric = input("Метрика [EI]: ") or "EI"
                    trajectories = np.array([h[metric] for h in multi_sim.all_histories])
                    
                    cumulative_means = []
                    for i in range(1, multi_sim.n_runs + 1):
                        subset = trajectories[:i, :]
                        cumulative_means.append(np.mean(subset, axis=0))
                    
                    final_mean = cumulative_means[-1]
                    errors = [np.mean(np.abs(mean - final_mean)) for mean in cumulative_means]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.suptitle(f'Анализ сходимости ({metric}): {multi_sim.get_params_string().replace("_", ", ")}')
                    
                    ax1.plot(range(1, multi_sim.n_runs + 1), errors, 'bo-', linewidth=2)
                    ax1.set_xlabel('Количество прогонов')
                    ax1.set_ylabel('Ошибка среднего')
                    ax1.set_title(f'Сходимость для {metric}')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_yscale('log')
                    
                    std_errors = []
                    for i in range(1, multi_sim.n_runs + 1):
                        subset = trajectories[:i, -1]
                        std_errors.append(np.std(subset))
                    
                    ax2.plot(range(1, multi_sim.n_runs + 1), std_errors, 'ro-', linewidth=2)
                    ax2.axhline(y=std_errors[-1], color='gray', linestyle='--', alpha=0.7, label='Финальное std')
                    ax2.set_xlabel('Количество прогонов')
                    ax2.set_ylabel('Стандартное отклонение')
                    ax2.set_title('Стабилизация разброса')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    
                    plt.tight_layout()
                    filename = f"convergence_{metric}_{multi_sim.get_params_string()}.png"
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"График сохранен: {filename}")
                    plt.show()
                    
                elif choice == "4":
                    run_idx = int(input(f"Номер прогона (1-{multi_sim.n_runs}) [1]: ") or "1") - 1
                    if 0 <= run_idx < multi_sim.n_runs:
                        model = multi_sim.runs[run_idx]
                        original_history = model.states_history
                        model.states_history = multi_sim.all_states_histories[run_idx]
                        model.plot_risk_heatmap()
                        model.states_history = original_history
                    else:
                        print("Неверный номер!")
                        
                elif choice == "0":
                    break
                    
    except KeyboardInterrupt:
        print("\n\nСимуляция прервана")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Основная функция"""
    print("Запуск программы симуляции...")
    print("Нажмите Ctrl+C для прерывания\n")
    
    try:
        # Проверяем, есть ли аргументы командной строки
        if len(sys.argv) > 1:
            run_simulation_with_params(auto_mode=True)
        else:
            run_simulation_with_params(auto_mode=False)
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена")
        sys.exit(0)


if __name__ == "__main__":
    main()
