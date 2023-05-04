import sys
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from scipy import stats
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtWidgets import QVBoxLayout
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np


class Modelo:
    def __init__(self):
        self.dataf = []
        
    def init_df(self, modulos):
        for modulo in range(modulos):
            self.dataf.append(None)

    def cargar_datos(self,n_modulo, df_fuente):
        self.dataf[n_modulo] = df_fuente

    def get_datos_ofilter(self,n_modulo ,x_min = 0,x_max = 0):
        columna_x = self.dataf[n_modulo].columns[0]
        if x_min == 0 and x_max == 0:
            return self.dataf[n_modulo]
        else:
            return self.dataf[n_modulo][(self.dataf[n_modulo][columna_x] >= x_min) & (self.dataf[n_modulo][columna_x] <= x_max)]

        return 
    
    def test_adf(self,df):
        # Realiza la prueba de Dickey-Fuller aumentada
        result = adfuller(df)
        return result

    def diff_df(self, n_modulo, diff_times):
        df_diff = self.get_datos_ofilter(n_modulo)

        columna_y = df_diff.columns[1]

        # Diferencia la columna especificada tantas veces como indica el parámetro diff
        for i in range(diff_times):
            df_diff[columna_y] = df_diff[columna_y].diff()
            df_diff = df_diff.dropna()

        return df_diff        

    def smoothexp_smooth_df (self,n_modulo, p_alpha): 
        df = self.get_datos_ofilter(n_modulo)

        col_x = df.columns[0]
        col_y = df.columns[1]
        # Aplica el alisado exponencial simple
        df['smooth_'+col_y] = df[col_y].ewm(alpha=p_alpha/100.).mean()
        return df

    def smoothexp_df_out (self, n_modulo, p_alpha): 
        df = self.get_datos_ofilter(n_modulo)
        col_x = df.columns[0]
        col_y = df.columns[1]
        # Aplica el alisado exponencial simple
        df[col_y] = df[col_y].ewm(alpha=p_alpha/100.).mean()
        return df

    
class Vista(QtWidgets.QMainWindow):        
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        # Cargar el archivo .ui
        uic.loadUi("pf01-main.ui", self)

        # Crear el QButtonGroup
        button_group = QButtonGroup()

        # Agregar los QRadioButton al QButtonGroup
        button_group.addButton(self.rb_grafico1)
        button_group.addButton(self.rb_grafico2)


        # Crear la figura y el eje
        self.figura_1 = Figure()
        self.ax_1 = self.figura_1.add_subplot(111)

        # Crear la figura y el eje
        self.figura_2 = Figure()
        self.ax_2 = self.figura_2.add_subplot(111)

        # Crear el FigureCanvas para mostrar la figura en el QWidget
        canvas_1 = FigureCanvas(self.figura_1)
        canvas_2 = FigureCanvas(self.figura_2)

        # Agregar el FigureCanvas al layout del QWidget
        self.grafico1.setLayout(QVBoxLayout())
        self.grafico1.layout().addWidget(canvas_1)
        self.grafico2.setLayout(QVBoxLayout())
        self.grafico2.layout().addWidget(canvas_2)
            

    def get_monitor_activo(self):
        if self.rb_grafico1.isChecked():
            return 1
        elif self.rb_grafico2.isChecked():
            return 2
        else:
            return 0

    def actualizar_grafico_2dxy(self, datos,col_x, col_y):
        print("#4 ")
        if(self.get_monitor_activo() == 1):
            print("#5 ")
            # Limpiar el eje
            self.ax_1.clear()

            # Dibujar el gráfico con los nuevos datos
            self.ax_1.plot(datos[col_x], datos[col_y])

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_1.canvas.draw()
        elif (self.get_monitor_activo() == 2):
                       
            # Limpiar el eje
            self.ax_2.clear()

            # Dibujar el gráfico con los nuevos datos
            self.ax_2.plot(datos[col_x], datos[col_y])

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_2.canvas.draw()
        else:
            pass

    def set_minmax_filterx(self, x_min, x_max):
        self.s_xmin.setValue(x_min)
        self.s_xmin.setMinimum(x_min)
        self.s_xmin.setMaximum(x_max)
        self.s_xmax.setValue(x_max)
        self.s_xmax.setMinimum(x_min)
        self.s_xmax.setMaximum( x_max)

    def get_minmax_filterx(self):##cambiar
        return self.s_xmin.value(), self.s_xmax.value()

    def set_minmax_labels_filterx(self):
        x_min,x_max = self.get_minmax_filterx()
        self.v_xmin.setText(str(x_min))
        self.v_xmax.setText(str(x_max))

    def get_alpha_smoothexp(self):##cambiar
        return self.s_smoothexp_alpha.value()
    
    def set_labels_smoothexp(self):
        s_alpha = self.get_alpha_smoothexp()
        self.v_smoothexp_alpha.setText(str(s_alpha))
        
    def write_console(self,line):
        self.consola1.appendPlainText(line)

    def actualizar_grafico_pacf(self,datos,col_x, col_y):
     
        if(self.get_monitor_activo() == 1):
          
            # Limpiar el eje
            self.ax_1.clear()

            # Dibujar el gráfico con los nuevos datos
            #self.ax_1.plot(datos[col_x], datos[col_y])
            plot_pacf(datos[col_y], ax=self.ax_1)

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_1.canvas.draw()
        elif (self.get_monitor_activo() == 2):
                       
            # Limpiar el eje
            self.ax_2.clear()
            
            plot_pacf(datos[col_y], ax=self.ax_2)

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_2.canvas.draw()
        else:
            pass

    def actualizar_grafico_qacf(self,datos,col_x, col_y):
     
        if(self.get_monitor_activo() == 1):
          
            # Limpiar el eje
            self.ax_1.clear()

            # Dibujar el gráfico con los nuevos datos
            #self.ax_1.plot(datos[col_x], datos[col_y])
            plot_acf(datos[col_y], ax=self.ax_1)

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_1.canvas.draw()
        elif (self.get_monitor_activo() == 2):
                       
            # Limpiar el eje
            self.ax_2.clear()
            
            plot_acf(datos[col_y], ax=self.ax_2)

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_2.canvas.draw()
        else:
            pass

    def actualizar_grafico_fit(self,train_data,test_data,predictions):

        predictions = np.insert(predictions, 0, train_data[-1])
        test_data = np.insert(test_data, 0, train_data[-1])

        if(self.get_monitor_activo() == 1):
            print("#5 ")
            # Limpiar el eje
            self.ax_1.clear()
   
   
            # Dibujar el gráfico con los nuevos datos
            self.ax_1.plot(train_data, label='Datos de entrenamiento')

            # Grafica los datos de prueba
            self.ax_1.plot(range(len(train_data)-1, len(train_data)+len(test_data)-1), test_data, label='Datos de prueba')

            # Grafica las predicciones
            self.ax_1.plot(range(len(train_data)-1, len(train_data)+len(test_data)-1), predictions, label='Predicciones')

            # Agrega una leyenda al gráfico
            self.ax_1.legend()

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_1.canvas.draw()
        elif (self.get_monitor_activo() == 2):
            print("#5 ")
            # Limpiar el eje
            self.ax_2.clear()

            # Dibujar el gráfico con los nuevos datos
            self.ax_2.plot(train_data, label='Datos de entrenamiento')

            # Grafica los datos de prueba
            self.ax_2.plot(range(len(train_data)-1, len(train_data)+len(test_data)-1), test_data, label='Datos de prueba')

            # Grafica las predicciones
            self.ax_2.plot(range(len(train_data)-1, len(train_data)+len(test_data)-1), predictions, label='Predicciones')

            # Agrega una leyenda al gráfico
            self.ax_2.legend()

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_2.canvas.draw()
        else:
            pass

    def actualizar_grafico_forecast(self,train_data,forecast):
        forecast = np.insert(forecast, 0, train_data[-1])

        if(self.get_monitor_activo() == 1):
            print("#5 ")
            # Limpiar el eje
            self.ax_1.clear()

            # Dibujar el gráfico con los nuevos datos
            self.ax_1.plot(train_data, label='Datos de entrenamiento')

            # Grafica las predicciones
            self.ax_1.plot(range(len(train_data)-1, len(train_data)+len(forecast)-1), forecast, label='Forecast')

            # Agrega una leyenda al gráfico
            self.ax_1.legend()

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_1.canvas.draw()
        elif (self.get_monitor_activo() == 2):
            print("#5 ")
            # Limpiar el eje
            self.ax_2.clear()

            # Dibujar el gráfico con los nuevos datos
            self.ax_2.plot(train_data, label='Datos de entrenamiento')

            # Grafica las predicciones
            self.ax_2.plot(range(len(train_data)-1, len(train_data)+len(forecast)-1), forecast, label='Forecast')

            # Agrega una leyenda al gráfico
            self.ax_2.legend()

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_2.canvas.draw()
        else:
            pass

    def actualizar_grafico_smoothexp(self, df, col_x, col_y):
        if(self.get_monitor_activo() == 1):
            print("#5 ")
            # Limpiar el eje
            self.ax_1.clear()

            # Visualiza los resultados
            self.ax_1.plot(df[col_x], df[col_y], label='Original')
            self.ax_1.plot(df[col_x], df['smooth_'+col_y], label='Smooth')

            # Agrega una leyenda al gráfico
            self.ax_1.legend()

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_1.canvas.draw()
        elif (self.get_monitor_activo() == 2):
            # Limpiar el eje
            self.ax_2.clear()

            # Visualiza los resultados
            self.ax_2.plot(df[col_x], df[col_y], label='Original')
            self.ax_2.plot(df[col_x], df['smooth_'+col_y], label='Smooth')

            # Agrega una leyenda al gráfico
            self.ax_2.legend()

            # Actualizar el FigureCanvas para mostrar los cambios
            self.figura_2.canvas.draw()
        else:
            pass

    def get_gui_val_text(self, nombre_control):
        control = getattr(self, nombre_control)
        seleccion = control.currentText()
        return seleccion

    def get_gui_val_num(self, nombre_control):
        control = getattr(self, nombre_control)
        seleccion = control.value()
        return seleccion    
        
class Controlador:
    def __init__(self, modelo, vista):
        self.modelo = modelo
        self.vista = vista

        self.modulo_indices = {
            'Loader': 0,
            'Filter X': 1,
            'ARIMA': 2,
            'Smooth Exp':3
        }

        self.init_seleccionar_modulo()

        ## Declaracion de eventos
        self.vista.l_modulos.currentItemChanged.connect(self.on_current_item_changed)
        self.vista.pb_loader_sel.clicked.connect(lambda: self.load_source_file('c_loader_ds','Loader'))  
        self.vista.pb_filterx_sel.clicked.connect(lambda: self.load_source_module('c_filterx_ds','Filter X'))
        self.vista.pb_smoothexp_sel.clicked.connect(lambda: self.load_source_module('c_smoothexp_ds','Smooth Exp'))
        self.vista.pb_arima_sel.clicked.connect(lambda: self.load_source_module('c_arima_ds','ARIMA'))

        self.vista.s_xmin.valueChanged.connect(self.refresh_grafico_filterx)

        self.vista.s_xmax.valueChanged.connect(self.refresh_grafico_filterx)
        
        self.vista.pb_arima_diff.clicked.connect(self.diff_serie_arima)   
        self.vista.pb_arima_pac.clicked.connect(self.show_pacf_arima)
        self.vista.pb_arima_auc.clicked.connect(self.show_qacf_arima)
        self.vista.pb_arima_fit.clicked.connect(self.calc_arima_fit)
        self.vista.pb_arima_forecast.clicked.connect(self.calc_arima_forecast)

        self.vista.pb_smoothexp_smooth.clicked.connect(self.calc_smoothexp_smooth)
        self.vista.s_smoothexp_alpha.valueChanged.connect(self.calc_smoothexp_smooth)
        

    def on_current_item_changed(self, current, previous):
        if current is not None:
            modulo_sel = current.text()
            self.cargar_modulo(modulo_sel)
        
    
    
    def init_seleccionar_modulo(self):
        modulo_sel = self.vista.l_modulos.currentItem().text()

        self.modelo.init_df(len(self.modulo_indices))
        
        self.cargar_modulo(modulo_sel)
        
    def cargar_modulo(self, modulo):
        indice = self.modulo_indices.get(modulo)
        if indice is not None:
            self.vista.s_controles.setCurrentIndex(indice)
        else:
            # Aquí puedes agregar el código para manejar opciones no especificadas en el diccionario
            print('Opción no encontrada')


    def get_module_n (self, nombre):
        return self.modulo_indices[nombre]
                      
    def load_source_file(self,fuente, modulo):

        df_fuente = self.sel_file_source(fuente)

        # Obtener el número correspondiente al nombre del módulo
        n_modulo = self.get_module_n(modulo) 
        

        self.modelo.cargar_datos( n_modulo, df_fuente )

        ##Post
        if modulo == 'Loader':
            # Obtener los nombres de las columnas 0 y 1
            col_x = df_fuente.columns[0]
            col_y = df_fuente.columns[1]

            # Llamar a la función actualizar_grafico con los nombres de las columnas
            self.vista.actualizar_grafico_2dxy (df_fuente,col_x, col_y)

    def load_source_module(self,fuente, modulo):
        src_modulo = self.vista.get_gui_val_text(fuente)

        df_fuente = None
        
        if src_modulo == 'Loader':
            df_fuente = self.modelo.get_datos_ofilter(self.get_module_n(src_modulo))
        elif src_modulo == 'Filter X':
            x_min,x_max = self.vista.get_minmax_filterx()
            df_fuente = self.modelo.get_datos_ofilter(self.get_module_n(src_modulo),x_min,x_max)
        elif src_modulo == "Smooth Exp":
            p_alpha = self.vista.get_alpha_smoothexp()
            n_modulo = self.get_module_n(src_modulo)
            df_fuente = self.modelo.smoothexp_df_out (n_modulo,p_alpha)
        else:
            pass        
        # Obtener el número correspondiente al nombre del módulo
        n_modulo = self.get_module_n(modulo) 
        self.modelo.cargar_datos( n_modulo, df_fuente)
  
        num_rows = df_fuente.shape[0]
        salida = "Module "+str(modulo)+" loaded with "+fuente+ ". Rows: "+str(num_rows)
        self.vista.write_console(salida)


        if modulo == 'Filter X':
            
            # Obtener los nombres de las columnas 0 y 1
            col_x = df_fuente.columns[0]
            col_y = df_fuente.columns[1]

            x_min = df_fuente[col_x].min()
            x_max = df_fuente[col_x].max()

            self.vista.set_minmax_filterx(x_min, x_max)
            
            self.refresh_grafico_filterx()
                    
        
    def refresh_grafico_filterx (self):

        x_min,x_max = self.vista.get_minmax_filterx()

        self.vista.set_minmax_labels_filterx()
        n_modulo = self.get_module_n('Filter X')
        df_grafico = self.modelo.get_datos_ofilter(n_modulo,x_min,x_max)
        print("#7 ",x_min,x_max)
        # Obtener los nombres de las columnas 0 y 1
        col_x = df_grafico.columns[0]
        col_y = df_grafico.columns[1]
        
        #Llamar a la función actualizar_grafico con los nombres de las columnas
        self.vista.actualizar_grafico_2dxy (df_grafico,col_x, col_y)

    def diff_serie_arima(self):

        diff = self.vista.get_gui_val_num('s_diff')
        n_modulo = self.get_module_n('ARIMA')

        df_diff = self.modelo.get_datos_ofilter(n_modulo)
        
        # Obtener los nombres de las columnas 0 y 1
        col_x = df_diff.columns[0]
        col_y = df_diff.columns[1]
        
        #test adf antes y a consola
        result = self.modelo.test_adf( df_diff[col_y] )

        adf = f'ADF Statistic: {result[0]}'
        p_value = f'p-value: {result[1]}'

        salida = "Diff 0 ARIMA. ADF "+str(adf)+ ". P-value: "+str(p_value)
        self.vista.write_console(salida)

        #diferenciar con diff
        df_diff = self.modelo.diff_df(n_modulo,diff) 

        # Obtener los nombres de las columnas 0 y 1
        col_x = df_diff.columns[0]
        col_y = df_diff.columns[1]
        
        #mostrar en la consola el adf
        result = self.modelo.test_adf( df_diff[col_y] )

        adf = f'ADF Statistic: {result[0]}'
        p_value = f'p-value: {result[1]}'

        salida = "Diff "+str(diff)+" ARIMA. ADF "+str(adf)+ ". P-value: "+str(p_value)
        self.vista.write_console(salida)


        
        #Llamar a la función actualizar_grafico con los nombres de las columnas
        self.vista.actualizar_grafico_2dxy (df_diff ,col_x, col_y) ##cambiar

    def show_pacf_arima(self):
        n_modulo = self.get_module_n('ARIMA')
        diff = self.vista.get_gui_val_num('s_diff')
        #diferenciar con diff
        df_diff = self.modelo.diff_df(n_modulo,diff)
        # Obtener los nombres de las columnas 0 y 1
        col_x = df_diff.columns[0]
        col_y = df_diff.columns[1]
        self.vista.actualizar_grafico_pacf(df_diff, col_x, col_y)
        
    def show_qacf_arima(self):
        n_modulo = self.get_module_n('ARIMA')
        diff = self.vista.get_gui_val_num('s_diff')
        #diferenciar con diff
        df_diff = self.modelo.diff_df(n_modulo,diff)
        # Obtener los nombres de las columnas 0 y 1
        col_x = df_diff.columns[0]
        col_y = df_diff.columns[1]
        self.vista.actualizar_grafico_qacf(df_diff, col_x, col_y)
        
    def calc_arima_fit(self):
        n_modulo = self.get_module_n('ARIMA')
        diff = self.vista.get_gui_val_num('s_diff')
        pac = self.vista.get_gui_val_num('s_pauto')
        qauc = self.vista.get_gui_val_num('s_qcorr')
        training_part = self.vista.get_gui_val_num('s_tdata')
        df_diff = self.modelo.diff_df(n_modulo,0)

        col_y = df_diff.columns[1]
        data = df_diff[col_y].values
        train_porc = training_part / 100.
        
        # Separa los datos en conjuntos de entrenamiento y prueba
        train_data = data[:int(train_porc*len(data))]
        test_data = data[int(train_porc*len(data)):]

        # Crea una instancia de la clase ARIMA
        model = ARIMA(data, order=(pac, diff, qauc))
        

        # Ajusta el modelo
        results = model.fit()

        # Realiza pronósticos
        predictions = results.predict(start=len(train_data), end=len(train_data)+len(test_data)-1,typ='levels')

        # Calcula el MSE
        mse = mean_squared_error(test_data, predictions)

        self.vista.write_console("fit mse "+str(mse))
        
        self.vista.actualizar_grafico_fit(train_data,test_data,predictions)        

    def calc_arima_forecast(self):

        n_modulo = self.get_module_n('ARIMA')
        diff = self.vista.get_gui_val_num('s_diff')
        pac = self.vista.get_gui_val_num('s_pauto')
        qauc = self.vista.get_gui_val_num('s_qcorr')
        f_steps = self.vista.get_gui_val_num('s_fsteps')
        df_diff = self.modelo.diff_df(n_modulo,0)

        col_y = df_diff.columns[1]
        data = df_diff[col_y].values

        # Crea una instancia de la clase ARIMA
        model = ARIMA(data, order=(pac, diff, qauc))

        # Ajusta el modelo
        model_fit = model.fit()

        # Hacemos predicciones fuera de la muestra en el conjunto de prueba
        forecast = model_fit.forecast(steps=f_steps)
        print("#9 ",forecast, f_steps, type(forecast))
        self.vista.actualizar_grafico_forecast(data,forecast)
        
    def calc_smoothexp_smooth(self):
        n_modulo = self.get_module_n('Smooth Exp')
        p_alpha = self.vista.get_alpha_smoothexp()
        self.vista.set_labels_smoothexp()
        df = self.modelo.smoothexp_smooth_df (n_modulo,p_alpha)
        col_x = df.columns[0]
        col_y = df.columns[1]
        self.vista.actualizar_grafico_smoothexp(df, col_x, col_y)
        
    def sel_file_source(self,n_modulo):

        df_ret = None
        # Obtener el texto de la selección actual del QComboBox
        seleccion = self.vista.get_gui_val_text(n_modulo)
        
        if seleccion == 'CSV file':
            file_name, _ = QFileDialog.getOpenFileName(self.vista, 'Seleccionar archivo CSV', '', 'CSV files (*.csv)')
            df_ret = pd.read_csv(file_name)
        elif seleccion == 'Sqlite3 file':
            file_name, _ = QFileDialog.getOpenFileName(self.vista, 'Seleccionar archivo SQLite3', '', 'SQLite3 files (*.sqlite3)')
        else:
            # Manejar otros casos aquí
            pass

        return df_ret
        
if __name__ == "__main__":
    #Controlador()
    app = QtWidgets.QApplication(sys.argv)

    model = Modelo()
    view = Vista(None)
    controller = Controlador(model, view)
    view.controller = controller

    view.show()

    sys.exit(app.exec_())
