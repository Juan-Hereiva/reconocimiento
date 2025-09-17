"""
Ventana de estadísticas y monitoreo en tiempo real
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import deque
import threading
import time
from datetime import datetime, timedelta

class StatsWindow:
    """Ventana de estadísticas y monitoreo"""
    
    def __init__(self, parent=None, face_recognition_system=None):
        self.parent = parent
        self.system = face_recognition_system
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("Estadísticas y Monitoreo del Sistema")
        self.window.geometry("900x700")
        self.window.configure(bg='#2c3e50')
        
        # Variables de monitoreo
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Datos históricos
        self.fps_history = deque(maxlen=100)
        self.detection_history = deque(maxlen=100)
        self.recognition_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        
        # Contadores
        self.total_frames = 0
        self.total_detections = 0
        self.total_recognitions = 0
        self.total_access_granted = 0
        self.total_access_denied = 0
        
        # Variables de interfaz
        self.stats_vars = {}
        self.init_stats_vars()
        
        # Crear interfaz
        self.create_widgets()
        
        # Iniciar monitoreo automático
        self.start_monitoring()
    
    def init_stats_vars(self):
        """Inicializar variables de estadísticas"""
        self.stats_vars = {
            'fps_current': tk.StringVar(value="0.0"),
            'fps_average': tk.StringVar(value="0.0"),
            'fps_max': tk.StringVar(value="0.0"),
            'total_frames': tk.StringVar(value="0"),
            'total_detections': tk.StringVar(value="0"),
            'total_recognitions': tk.StringVar(value="0"),
            'access_granted': tk.StringVar(value="0"),
            'access_denied': tk.StringVar(value="0"),
            'detection_rate': tk.StringVar(value="0.0%"),
            'recognition_rate': tk.StringVar(value="0.0%"),
            'success_rate': tk.StringVar(value="0.0%"),
            'uptime': tk.StringVar(value="00:00:00"),
            'database_persons': tk.StringVar(value="0"),
            'last_recognition': tk.StringVar(value="Ninguno"),
            'system_status': tk.StringVar(value="Detenido")
        }
        
        self.start_time = datetime.now()
    
    def create_widgets(self):
        """Crear widgets de la interfaz"""
        # Notebook para pestañas
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña de estadísticas en tiempo real
        self.create_realtime_tab(notebook)
        
        # Pestaña de gráficos
        self.create_charts_tab(notebook)
        
        # Pestaña de base de datos
        self.create_database_tab(notebook)
        
        # Pestaña de logs
        self.create_logs_tab(notebook)
    
    def create_realtime_tab(self, parent):
        """Crear pestaña de estadísticas en tiempo real"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Tiempo Real")
        
        # Frame principal
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sección de rendimiento
        perf_frame = ttk.LabelFrame(main_frame, text="Rendimiento del Sistema", padding=10)
        perf_frame.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 5))
        
        self.create_stat_row(perf_frame, "FPS Actual:", self.stats_vars['fps_current'])
        self.create_stat_row(perf_frame, "FPS Promedio:", self.stats_vars['fps_average'])
        self.create_stat_row(perf_frame, "FPS Máximo:", self.stats_vars['fps_max'])
        self.create_stat_row(perf_frame, "Tiempo Activo:", self.stats_vars['uptime'])
        self.create_stat_row(perf_frame, "Estado Sistema:", self.stats_vars['system_status'])
        
        # Sección de detección
        detection_frame = ttk.LabelFrame(main_frame, text="Detección y Reconocimiento", padding=10)
        detection_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 5))
        
        self.create_stat_row(detection_frame, "Frames Totales:", self.stats_vars['total_frames'])
        self.create_stat_row(detection_frame, "Detecciones:", self.stats_vars['total_detections'])
        self.create_stat_row(detection_frame, "Reconocimientos:", self.stats_vars['total_recognitions'])
        self.create_stat_row(detection_frame, "Tasa Detección:", self.stats_vars['detection_rate'])
        self.create_stat_row(detection_frame, "Tasa Reconocimiento:", self.stats_vars['recognition_rate'])
        
        # Sección de acceso
        access_frame = ttk.LabelFrame(main_frame, text="Control de Acceso", padding=10)
        access_frame.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=(5, 0))
        
        self.create_stat_row(access_frame, "Accesos Permitidos:", self.stats_vars['access_granted'])
        self.create_stat_row(access_frame, "Accesos Denegados:", self.stats_vars['access_denied'])
        self.create_stat_row(access_frame, "Tasa de Éxito:", self.stats_vars['success_rate'])
        self.create_stat_row(access_frame, "Último Reconocimiento:", self.stats_vars['last_recognition'])
        
        # Sección de base de datos
        db_frame = ttk.LabelFrame(main_frame, text="Base de Datos", padding=10)
        db_frame.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        
        self.create_stat_row(db_frame, "Personas Registradas:", self.stats_vars['database_persons'])
        
        # Botones de control
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(control_frame, text="Reiniciar Estadísticas", 
                  command=self.reset_stats).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Exportar Datos", 
                  command=self.export_stats).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Actualizar", 
                  command=self.update_stats).pack(side=tk.LEFT, padx=5)
        
        # Configurar grid
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
    
    def create_charts_tab(self, parent):
        """Crear pestaña de gráficos"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Gráficos")
        
        # Crear figura de matplotlib
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor('#34495e')
        
        # Configurar estilo de gráficos
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        # Canvas de matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurar gráficos
        self.setup_charts()
    
    def create_database_tab(self, parent):
        """Crear pestaña de base de datos"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Base de Datos")
        
        # Información de personas
        persons_frame = ttk.LabelFrame(frame, text="Personas Registradas", padding=10)
        persons_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview para mostrar personas
        columns = ('Nombre', 'Embeddings', 'Fecha Registro', 'Último Acceso')
        self.persons_tree = ttk.Treeview(persons_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.persons_tree.heading(col, text=col)
            self.persons_tree.column(col, width=150)
        
        # Scrollbar para treeview
        scrollbar_persons = ttk.Scrollbar(persons_frame, orient=tk.VERTICAL, command=self.persons_tree.yview)
        self.persons_tree.configure(yscrollcommand=scrollbar_persons.set)
        
        self.persons_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_persons.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones de base de datos
        db_buttons_frame = ttk.Frame(frame)
        db_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(db_buttons_frame, text="Actualizar Lista", 
                  command=self.update_persons_list).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(db_buttons_frame, text="Ver Detalles", 
                  command=self.show_person_details).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(db_buttons_frame, text="Backup Base de Datos", 
                  command=self.backup_database).pack(side=tk.LEFT, padx=5)
    
    def create_logs_tab(self, parent):
        """Crear pestaña de logs"""
        frame = ttk.Frame(parent)
        parent.add(frame, text="Logs del Sistema")
        
        # Área de texto para logs
        log_frame = ttk.LabelFrame(frame, text="Registro de Eventos", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget con scrollbar
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=20, 
                               bg='#2c3e50', fg='white', font=('Consolas', 10))
        
        scrollbar_log = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar_log.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_log.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones de logs
        log_buttons_frame = ttk.Frame(frame)
        log_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(log_buttons_frame, text="Limpiar Logs", 
                  command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(log_buttons_frame, text="Guardar Logs", 
                  command=self.save_logs).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(log_buttons_frame, text="Auto-scroll", 
                  command=self.toggle_autoscroll).pack(side=tk.LEFT, padx=5)
        
        self.autoscroll = True
        
        # Cargar logs iniciales
        self.load_initial_logs()
    
    def create_stat_row(self, parent, label, textvariable):
        """Crear fila de estadística"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label, width=20).pack(side=tk.LEFT)
        ttk.Label(frame, textvariable=textvariable, width=15, 
                 font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
    
    def setup_charts(self):
        """Configurar gráficos iniciales"""
        # Gráfico 1: FPS en tiempo real
        self.ax1.set_title('FPS en Tiempo Real', color='white', fontsize=12)
        self.ax1.set_xlabel('Tiempo', color='white')
        self.ax1.set_ylabel('FPS', color='white')
        self.line1, = self.ax1.plot([], [], 'g-', linewidth=2)
        
        # Gráfico 2: Tasa de detección
        self.ax2.set_title('Tasa de Detección', color='white', fontsize=12)
        self.ax2.set_xlabel('Tiempo', color='white')
        self.ax2.set_ylabel('Detecciones/min', color='white')
        self.line2, = self.ax2.plot([], [], 'b-', linewidth=2)
        
        # Gráfico 3: Tasa de reconocimiento
        self.ax3.set_title('Tasa de Reconocimiento', color='white', fontsize=12)
        self.ax3.set_xlabel('Tiempo', color='white')
        self.ax3.set_ylabel('Reconocimientos/min', color='white')
        self.line3, = self.ax3.plot([], [], 'r-', linewidth=2)
        
        # Gráfico 4: Distribución de accesos (pie chart)
        self.ax4.set_title('Distribución de Accesos', color='white', fontsize=12)
        
        plt.tight_layout()
    
    def start_monitoring(self):
        """Iniciar monitoreo del sistema"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Detener monitoreo del sistema"""
        self.is_monitoring = False
    
    def monitor_loop(self):
        """Loop principal de monitoreo"""
        while self.is_monitoring:
            try:
                self.update_stats()
                self.update_charts()
                time.sleep(1)  # Actualizar cada segundo
            except Exception as e:
                self.add_log(f"Error en monitoreo: {e}", "ERROR")
    
    def update_stats(self):
        """Actualizar estadísticas"""
        try:
            current_time = datetime.now()
            
            # Calcular uptime
            uptime = current_time - self.start_time
            uptime_str = str(uptime).split('.')[0]  # Remover microsegundos
            self.stats_vars['uptime'].set(uptime_str)
            
            # Si hay sistema conectado, obtener estadísticas reales
            if self.system:
                # FPS
                fps = getattr(self.system, 'fps', 0.0)
                self.stats_vars['fps_current'].set(f"{fps:.1f}")
                
                if fps > 0:
                    self.fps_history.append(fps)
                    self.time_history.append(current_time)
                    
                    avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                    max_fps = np.max(self.fps_history) if self.fps_history else 0
                    
                    self.stats_vars['fps_average'].set(f"{avg_fps:.1f}")
                    self.stats_vars['fps_max'].set(f"{max_fps:.1f}")
                
                # Estado del sistema
                status = "Ejecutándose" if getattr(self.system, 'is_running', False) else "Detenido"
                self.stats_vars['system_status'].set(status)
                
                # Estadísticas de detección (simuladas por ahora)
                self.total_frames += 1
                self.stats_vars['total_frames'].set(str(self.total_frames))
                
                # Simular detecciones y reconocimientos
                if np.random.random() > 0.3:  # 70% de detecciones
                    self.total_detections += 1
                    
                    if np.random.random() > 0.5:  # 50% de reconocimientos exitosos
                        self.total_recognitions += 1
                        
                        if np.random.random() > 0.7:  # 30% de accesos permitidos
                            self.total_access_granted += 1
                        else:
                            self.total_access_denied += 1
                
                # Actualizar variables
                self.stats_vars['total_detections'].set(str(self.total_detections))
                self.stats_vars['total_recognitions'].set(str(self.total_recognitions))
                self.stats_vars['access_granted'].set(str(self.total_access_granted))
                self.stats_vars['access_denied'].set(str(self.total_access_denied))
                
                # Calcular tasas
                if self.total_frames > 0:
                    detection_rate = (self.total_detections / self.total_frames) * 100
                    self.stats_vars['detection_rate'].set(f"{detection_rate:.1f}%")
                
                if self.total_detections > 0:
                    recognition_rate = (self.total_recognitions / self.total_detections) * 100
                    self.stats_vars['recognition_rate'].set(f"{recognition_rate:.1f}%")
                
                total_access = self.total_access_granted + self.total_access_denied
                if total_access > 0:
                    success_rate = (self.total_access_granted / total_access) * 100
                    self.stats_vars['success_rate'].set(f"{success_rate:.1f}%")
                
                # Base de datos
                if hasattr(self.system, 'face_recognizer'):
                    db_count = len(self.system.face_recognizer.face_database)
                    self.stats_vars['database_persons'].set(str(db_count))
            
        except Exception as e:
            self.add_log(f"Error actualizando estadísticas: {e}", "ERROR")
    
    def update_charts(self):
        """Actualizar gráficos"""
        try:
            if len(self.fps_history) > 1:
                # Actualizar gráfico de FPS
                times = list(range(len(self.fps_history)))
                self.line1.set_data(times, list(self.fps_history))
                self.ax1.relim()
                self.ax1.autoscale_view()
                
                # Actualizar gráfico de detecciones (simulado)
                detection_rates = [len(self.fps_history) * 0.7] * len(self.fps_history)  # Simulado
                self.line2.set_data(times, detection_rates)
                self.ax2.relim()
                self.ax2.autoscale_view()
                
                # Actualizar gráfico de reconocimientos (simulado)
                recognition_rates = [len(self.fps_history) * 0.35] * len(self.fps_history)  # Simulado
                self.line3.set_data(times, recognition_rates)
                self.ax3.relim()
                self.ax3.autoscale_view()
                
                # Actualizar pie chart de accesos
                self.ax4.clear()
                if self.total_access_granted > 0 or self.total_access_denied > 0:
                    labels = ['Permitidos', 'Denegados']
                    sizes = [self.total_access_granted, self.total_access_denied]
                    colors = ['#27ae60', '#e74c3c']
                    
                    self.ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                               textprops={'color': 'white'})
                    self.ax4.set_title('Distribución de Accesos', color='white', fontsize=12)
                
                # Actualizar canvas
                self.canvas.draw()
        
        except Exception as e:
            print(f"Error actualizando gráficos: {e}")
    
    def reset_stats(self):
        """Reiniciar estadísticas"""
        self.total_frames = 0
        self.total_detections = 0
        self.total_recognitions = 0
        self.total_access_granted = 0
        self.total_access_denied = 0
        
        self.fps_history.clear()
        self.detection_history.clear()
        self.recognition_history.clear()
        self.time_history.clear()
        
        self.start_time = datetime.now()
        
        self.add_log("Estadísticas reiniciadas", "INFO")
    
    def export_stats(self):
        """Exportar estadísticas a archivo"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stats_export_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("ESTADÍSTICAS DEL SISTEMA DE RECONOCIMIENTO FACIAL\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for key, var in self.stats_vars.items():
                    f.write(f"{key}: {var.get()}\n")
                
                f.write(f"\nHistorial FPS: {list(self.fps_history)}\n")
            
            self.add_log(f"Estadísticas exportadas a {filename}", "INFO")
            
        except Exception as e:
            self.add_log(f"Error exportando estadísticas: {e}", "ERROR")
    
    def update_persons_list(self):
        """Actualizar lista de personas en la base de datos"""
        try:
            # Limpiar treeview
            for item in self.persons_tree.get_children():
                self.persons_tree.delete(item)
            
            # Si hay sistema conectado, obtener datos reales
            if self.system and hasattr(self.system, 'database_manager'):
                persons = self.system.database_manager.list_persons()
                
                for person_name in persons:
                    person_data = self.system.database_manager.get_person(person_name)
                    if person_data:
                        embeddings, metadata = person_data
                        
                        num_embeddings = metadata.get('num_embeddings', 1)
                        date_added = metadata.get('added_date', 'Desconocido')
                        if date_added != 'Desconocido':
                            try:
                                # Formatear fecha
                                date_obj = datetime.fromisoformat(date_added.replace('Z', '+00:00'))
                                date_added = date_obj.strftime('%Y-%m-%d %H:%M')
                            except:
                                pass
                        
                        last_access = metadata.get('last_access', 'Nunca')
                        
                        self.persons_tree.insert('', 'end', values=(
                            person_name, num_embeddings, date_added, last_access
                        ))
            
        except Exception as e:
            self.add_log(f"Error actualizando lista de personas: {e}", "ERROR")
    
    def show_person_details(self):
        """Mostrar detalles de persona seleccionada"""
        selection = self.persons_tree.selection()
        if not selection:
            return
        
        item = self.persons_tree.item(selection[0])
        person_name = item['values'][0]
        
        # Crear ventana de detalles
        details_window = tk.Toplevel(self.window)
        details_window.title(f"Detalles: {person_name}")
        details_window.geometry("400x300")
        
        # Mostrar información detallada
        info_text = tk.Text(details_window, wrap=tk.WORD, padx=10, pady=10)
        info_text.pack(fill=tk.BOTH, expand=True)
        
        try:
            if self.system and hasattr(self.system, 'database_manager'):
                person_data = self.system.database_manager.get_person(person_name)
                if person_data:
                    embeddings, metadata = person_data
                    
                    info_text.insert(tk.END, f"INFORMACIÓN DE: {person_name}\n")
                    info_text.insert(tk.END, "=" * 30 + "\n\n")
                    
                    for key, value in metadata.items():
                        info_text.insert(tk.END, f"{key.replace('_', ' ').title()}: {value}\n")
                    
                    info_text.insert(tk.END, f"\nForma del embedding: {embeddings.shape}\n")
                    info_text.insert(tk.END, f"Tipo de dato: {embeddings.dtype}\n")
        
        except Exception as e:
            info_text.insert(tk.END, f"Error cargando detalles: {e}")
        
        info_text.config(state=tk.DISABLED)
    
    def backup_database(self):
        """Crear backup de la base de datos"""
        try:
            if self.system and hasattr(self.system, 'database_manager'):
                success = self.system.database_manager.backup_database()
                if success:
                    self.add_log("Backup de base de datos creado exitosamente", "INFO")
                else:
                    self.add_log("Error creando backup de base de datos", "ERROR")
        except Exception as e:
            self.add_log(f"Error en backup: {e}", "ERROR")
    
    def add_log(self, message, level="INFO"):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        
        if self.autoscroll:
            self.log_text.see(tk.END)
    
    def clear_logs(self):
        """Limpiar logs"""
        self.log_text.delete(1.0, tk.END)
        self.add_log("Logs limpiados", "INFO")
    
    def save_logs(self):
        """Guardar logs a archivo"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_logs_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            
            self.add_log(f"Logs guardados en {filename}", "INFO")
            
        except Exception as e:
            self.add_log(f"Error guardando logs: {e}", "ERROR")
    
    def toggle_autoscroll(self):
        """Alternar auto-scroll"""
        self.autoscroll = not self.autoscroll
        status = "habilitado" if self.autoscroll else "deshabilitado"
        self.add_log(f"Auto-scroll {status}", "INFO")
    
    def load_initial_logs(self):
        """Cargar logs iniciales"""
        self.add_log("Sistema de monitoreo iniciado", "INFO")
        self.add_log("Ventana de estadísticas abierta", "INFO")
        
        if self.system:
            self.add_log("Sistema de reconocimiento facial conectado", "INFO")
        else:
            self.add_log("Sistema de reconocimiento facial no conectado", "WARNING")


if __name__ == "__main__":
    # Prueba independiente
    stats_window = StatsWindow()
    stats_window.window.mainloop()