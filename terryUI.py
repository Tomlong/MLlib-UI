import wx
import  cStringIO
from mlKmeans import KMeansModel
from mlsvm import SVMModel
from mllogisticRegression import LogisticRegressionModel
from mlNaive import NaiveBayesModel
from mlLasso import LassoModel
from mlLinearRegression import LinearRegressionModel
from mlRidgeRegression import RidgeRegressionModel
class bucky(wx.Frame):

    def __init__(self, parent, id):
        # initial a Frame 
        wx.Frame.__init__(self, parent, id, "Spark-Mllib", size = (600,50))
        # 200 = width, 300 = tall
        panel = wx.Panel(self)

        # initial algorithm selection
        self.algorithm = -1
        # initial Machine Learning selection
        self.catalog_choice = wx.Choice(panel, id, (-1, -1), (-1, -1), ["---Machine Learning---","Clustering", "Classification", "Regression"], name = "Machine Learning")
        # initial Algorithm selection
        self.algorithm_choice = wx.Choice(panel, id, (-1, -1), (-1, -1), ["---Algorithm---"], name = "algorithm")

        # Algorithm choosing button
        checkButton = wx.Button(panel, label = "Check")
        # setting button's Font
        font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        checkButton.SetFont(font)
        
        # Setting the layout of main menu 
        # Using BoxSizer (Horizontal)
        bs_main = wx.BoxSizer(wx.HORIZONTAL)
        bs_main.AddSpacer(40) 
        bs_main.Add(self.catalog_choice, 1) 
        bs_main.Add(self.algorithm_choice, 1)
        bs_main.AddSpacer(60) 
        bs_main.Add(checkButton, 1)
        bs_main.AddSpacer(40)
        #put the sizer on panel 
        panel.SetSizer(bs_main)
    
        # After checking, it will show the selected algorithm frame.
        self.Bind(wx.EVT_BUTTON, self.onCheck, checkButton)
        self.Bind(wx.EVT_CHOICE, self.onChoiceCatalog, self.catalog_choice)
    

    def onCheck(self, event):
        print self.algorithm
        # KMeans
        if self.algorithm == 0:
            frame = KMeans(parent = None, id = 1)
            frame.Show()

        # SVM
        elif self.algorithm == 1:
            frame = classification(parent = None, id = 1, catalog = 0)
            frame.Show()

        # Logistic Regression    
        elif self.algorithm == 2:
            frame = classification(parent = None, id = 1, catalog = 1)
            frame.Show()

        # Naive Bayes
        elif self.algorithm == 3:
            frame = classification(parent = None, id = 1, catalog = 2)
            frame.Show()  

        # Lasso
        elif self.algorithm == 4:
            frame = regression(parent = None, id = 1, catalog = 0)
            frame.Show()

        # Linear
        elif self.algorithm == 5:
            frame = regression(parent = None, id = 1, catalog = 1)
            frame.Show()

        # Ridge
        elif self.algorithm == 6:
            frame = regression(parent = None, id = 1, catalog = 2)
            frame.Show()   


    def onChoiceCatalog(self, event):

        #Dynamic changing the algorithm choice

        if self.catalog_choice.GetCurrentSelection() == 0:
            self.algorithm_choice.SetItems(["---Algorithm---"])

        elif self.catalog_choice.GetCurrentSelection() == 1:
            self.algorithm_choice.SetItems(["---Algorithm---","Kmeans"])
            self.Bind(wx.EVT_CHOICE, self.onCheckClustering, self.algorithm_choice)
            
        elif self.catalog_choice.GetCurrentSelection() == 2:
            self.algorithm_choice.SetItems(["---Algorithm---","SVM", "Logistic Regression", "NaiveBayes"])            
            self.Bind(wx.EVT_CHOICE, self.onCheckClassification, self.algorithm_choice)

        elif self.catalog_choice.GetCurrentSelection() == 3:
            self.algorithm_choice.SetItems(["---Algorithm---","Lasso", "Linear", "Ridge"])           
            self.Bind(wx.EVT_CHOICE, self.onCheckRegression, self.algorithm_choice)

    def onCheckClustering(self, event):

        #Choosing on the wrong selection
        if self.algorithm_choice.GetCurrentSelection() == 0:
            self.algorithm = -1

        #Choosing on the KMeans
        elif self.algorithm_choice.GetCurrentSelection() == 1:
            self.algorithm = 0

    def onCheckClassification(self, event):

        #Choosing on the wrong selection
        if self.algorithm_choice.GetCurrentSelection() == 0:
            self.algorithm = -1

        #Choosing on the SVM
        elif self.algorithm_choice.GetCurrentSelection() == 1:
            self.algorithm = 1

        #Choosing on the Logistic Regression
        elif self.algorithm_choice.GetCurrentSelection() == 2:
            self.algorithm = 2

        #Choosing on the Naive Bayes
        elif self.algorithm_choice.GetCurrentSelection() == 3:
            self.algorithm = 3

    def onCheckRegression(self, event):

        #Choosing on the wrong selection
        if self.algorithm_choice.GetCurrentSelection() == 0:
            self.algorithm = -1

        #Choosing on the Lasso
        elif self.algorithm_choice.GetCurrentSelection() == 1:
            self.algorithm = 4

        #Choosing on the Linear  
        elif self.algorithm_choice.GetCurrentSelection() == 2:
            self.algorithm = 5

        #Choosing on the Ridge
        elif self.algorithm_choice.GetCurrentSelection() == 3:
            self.algorithm = 6
    

class KMeans(wx.Frame):
        
    def __init__(self, parent, id):

        # Initialize

        wx.Frame.__init__(self, parent, id, "KMeans", size = (600, 700))
        self.panel = wx.Panel(self) 
        self.id = id 
        self.dataPath = ""
        self.readFirst = True        
        
        # Add the hbox
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox_select = wx.BoxSizer(wx.HORIZONTAL)            
        hbox_Kcluster = wx.BoxSizer(wx.HORIZONTAL)            
        hbox_label = wx.BoxSizer(wx.HORIZONTAL)
        hbox_split = wx.BoxSizer(wx.HORIZONTAL) 
        hbox_master = wx.BoxSizer(wx.HORIZONTAL)           
        hbox_start = wx.BoxSizer(wx.HORIZONTAL)            
        hbox_result = wx.BoxSizer(wx.HORIZONTAL)
    
        #Data Selection
        selectButton = wx.Button(self.panel, label = "Data")
        font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        selectButton.SetFont(font)
        self.textName = wx.StaticText(self.panel, id, "", style = wx.ALIGN_LEFT)        
        hbox_select.AddSpacer(20)
        hbox_select.Add(selectButton, 1)
        hbox_select.Add(self.textName, 3)
        
        #Kclusters
        Kcluster_text = wx.StaticText(self.panel, id, "Kcluster(s): ",style = wx.ALIGN_LEFT)
        self.Kcluster_enter = wx.TextCtrl(self.panel, id, size = (30, -1), style = wx.TE_PROCESS_TAB)
        hbox_Kcluster.AddSpacer(20)
        hbox_Kcluster.Add(Kcluster_text, 1)
        hbox_Kcluster.Add(self.Kcluster_enter)        
        
        #Location
        label_text = wx.StaticText(self.panel, id, "Label(location): ", style = wx.ALIGN_LEFT)
        self.label_choice = wx.Choice(self.panel, id, (-1, -1), (-1, -1), ['First', 'End'])        
        hbox_label.AddSpacer(20)
        hbox_label.Add(label_text, 1)
        hbox_label.Add(self.label_choice, 1)  
      
        #Split Character
        split_text = wx.StaticText(self.panel, id, "Split Character: ", style = wx.ALIGN_LEFT)
        self.split_enter = wx.TextCtrl(self.panel, id, size = (20, -1), style = wx.TE_PROCESS_TAB)
        hbox_split.AddSpacer(20)
        hbox_split.Add(split_text, 1)
        hbox_split.Add(self.split_enter)
        
        #Master
        master_text = wx.StaticText(self.panel, id, "Master: ", style = wx.ALIGN_LEFT)
        self.master_enter = wx.TextCtrl(self.panel, id, size = (50, -1), style = wx.TE_PROCESS_TAB)
        hbox_master.AddSpacer(20)
        hbox_master.Add(master_text, 1)
        hbox_master.Add(self.master_enter)

        checkButton = wx.Button(self.panel, label = "Start")
        font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        
        checkButton.SetFont(font)        
        hbox_start.AddSpacer(20)
        hbox_start.Add(checkButton, 1)
        
        #result 
        self.result = wx.TextCtrl(self.panel, id, style = wx.TE_READONLY | wx.TE_MULTILINE)       
        hbox_result.Add(self.result, 1, wx.EXPAND)
        
        # Default path for openning data
        self.dirname = '/Users/Terry/Work/spark-1.0.1-bin-hadoop1/project'
        
        # Box setting
        vbox.AddSpacer(10)
        vbox.Add(hbox_select, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_Kcluster, flag=wx.LEFT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_label, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_split, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_master, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_start, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_result, 1, flag = wx.EXPAND | wx.LEFT | wx.RIGHT, border = 20)
        vbox.Add((-1, 20))
        self.panel.SetSizer(vbox)
        

        # press on data button
        self.Bind(wx.EVT_BUTTON,self.selectData, selectButton)
        # press on start button
        self.Bind(wx.EVT_BUTTON, self.start, checkButton)

    def start(self, event):
        id = self.id       
        panel = self.panel
        information = True        

        if (self.Kcluster_enter.IsEmpty() | self.master_enter.IsEmpty()):
            information = False
        
        if information:
            k = int(self.Kcluster_enter.GetValue())
            label = int(self.label_choice.GetCurrentSelection())
            if self.split_enter.IsEmpty():
                character = ' '
            else:
                character = self.split_enter.GetValue()
            master = self.master_enter.GetValue()
            string = KMeansModel(self.dataPath, label, k, character, master)
        
        else:
            string = "The information is not complete.\nPlease check again."
        self.result.SetValue(string)
        #print model.centers
        
        #text1 = wx.StaticText(panel, id, k, (10, 100), (-1, -1), wx.ALIGN_LEFT)
    def selectData(self, event):
        id = self.id       
        panel = self.panel
        clear = " " * 100
        dig = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.*", wx.OPEN)

        if dig.ShowModal() == wx.ID_OK:
            filename = dig.GetFilename()
            path = dig.GetDirectory()
            self.dataPath = path + '/' + filename
            self.textName.SetLabel(filename)
            #Clean the information
            self.Kcluster_enter.SetValue("")
            self.split_enter.SetValue("")
            self.master_enter.SetValue("")        
            self.result.SetValue("")

class classification(wx.Frame):
        
    def __init__(self, parent, id, catalog):

        self.catalog = catalog
        if self.catalog == 0:
            wx.Frame.__init__(self, parent, id, "SVM", size = (600, 700))
        elif self.catalog == 1:
            wx.Frame.__init__(self, parent, id, "Logistic Regression", size = (600, 700))
        elif self.catalog == 2:
            wx.Frame.__init__(self, parent, id, "Naive Bayes", size = (600, 700))

        self.panel = wx.Panel(self) 
        self.id = id 
        self.dataPath = ""
        self.readFirst = True        
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox_select = wx.BoxSizer(wx.HORIZONTAL)            
        hbox_min_max = wx.BoxSizer(wx.HORIZONTAL)            
        hbox_label = wx.BoxSizer(wx.HORIZONTAL)
        hbox_split = wx.BoxSizer(wx.HORIZONTAL) 
        hbox_normalize = wx.BoxSizer(wx.HORIZONTAL)
        hbox_pca = wx.BoxSizer(wx.HORIZONTAL)
        hbox_master = wx.BoxSizer(wx.HORIZONTAL)           
        hbox_start = wx.BoxSizer(wx.HORIZONTAL)            
        hbox_result = wx.BoxSizer(wx.HORIZONTAL)
    
        #Data Selection
        selectButton = wx.Button(self.panel, label = "Data")
        font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        selectButton.SetFont(font)
        self.textName = wx.StaticText(self.panel, id, "", style = wx.ALIGN_LEFT)        
        hbox_select.AddSpacer(20)
        hbox_select.Add(selectButton, 1)
        hbox_select.Add(self.textName, 3)
        
        #max & min label
        if self.catalog != 2:
            maxLabel_text = wx.StaticText(self.panel, id, "max label: ",style = wx.ALIGN_LEFT)
            self.maxLabel_enter = wx.TextCtrl(self.panel, id, size = (30, -1), style = wx.TE_PROCESS_TAB)
            minLabel_text = wx.StaticText(self.panel, id, "min label: ",style = wx.ALIGN_LEFT)
            self.minLabel_enter = wx.TextCtrl(self.panel, id, size = (30, -1), style = wx.TE_PROCESS_TAB)
            hbox_min_max.AddSpacer(20)
            hbox_min_max.Add(maxLabel_text, 1)
            hbox_min_max.Add(self.maxLabel_enter)   
            hbox_min_max.Add(minLabel_text, 1)
            hbox_min_max.Add(self.minLabel_enter)  

        #Normalization
        normalize_text =  wx.StaticText(self.panel, id, "Normalization (min-max): ", style = wx.ALIGN_LEFT)
        self.normalize_choice = wx.Choice(self.panel, id, (-1, -1), (-1, -1), ['No', 'Yes'])        
        hbox_normalize.AddSpacer(20)
        hbox_normalize.Add(normalize_text, 2)
        hbox_normalize.Add(self.normalize_choice, 1)

        #PCA
        pca_text =  wx.StaticText(self.panel, id, "PCA: ", style = wx.ALIGN_LEFT)
        self.pca_choice = wx.Choice(self.panel, id, (-1, -1), (-1, -1), ['No', 'Yes'])        
        hbox_pca.AddSpacer(20)
        hbox_pca.Add(pca_text, 1)
        hbox_pca.Add(self.pca_choice, 1)    
        
        #Location
        label_text = wx.StaticText(self.panel, id, "Label(location): ", style = wx.ALIGN_LEFT)
        self.label_choice = wx.Choice(self.panel, id, (-1, -1), (-1, -1), ['First', 'End'])        
        hbox_label.AddSpacer(20)
        hbox_label.Add(label_text, 1)
        hbox_label.Add(self.label_choice, 1)  
      
        #Split Character
        split_text = wx.StaticText(self.panel, id, "Split Character: ", style = wx.ALIGN_LEFT)
        self.split_enter = wx.TextCtrl(self.panel, id, size = (20, -1), style = wx.TE_PROCESS_TAB)
        hbox_split.AddSpacer(20)
        hbox_split.Add(split_text, 1)
        hbox_split.Add(self.split_enter)
        
        #Master
        master_text = wx.StaticText(self.panel, id, "Master: ", style = wx.ALIGN_LEFT)
        self.master_enter = wx.TextCtrl(self.panel, id, size = (50, -1), style = wx.TE_PROCESS_TAB)
        hbox_master.AddSpacer(20)
        hbox_master.Add(master_text, 1)
        hbox_master.Add(self.master_enter)

        checkButton = wx.Button(self.panel, label = "Start")
        font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        
        checkButton.SetFont(font)        
        hbox_start.AddSpacer(20)
        hbox_start.Add(checkButton, 1)
        
        #result 
        self.result = wx.TextCtrl(self.panel, id, style = wx.TE_READONLY | wx.TE_MULTILINE)       
        hbox_result.Add(self.result, 1, wx.EXPAND)
        
        # Default path for openning data
        self.dirname = '/Users/Terry/Work/spark-1.0.1-bin-hadoop1/project'
        
        # Box setting
        vbox.AddSpacer(10)
        vbox.Add(hbox_select, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        if self.catalog != 2:
            vbox.Add(hbox_min_max, flag=wx.LEFT|wx.TOP)
            vbox.Add((-1, 10))
        vbox.Add(hbox_label, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_normalize, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_pca, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_split, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_master, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_start, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_result, 1, flag = wx.EXPAND | wx.LEFT | wx.RIGHT, border = 20)
        vbox.Add((-1, 20))
        self.panel.SetSizer(vbox)
        

        # button event
        self.Bind(wx.EVT_BUTTON,self.selectData, selectButton)
        self.Bind(wx.EVT_BUTTON, self.start, checkButton)

    def start(self, event):
        id = self.id       
        panel = self.panel
        information = True        

        if (self.master_enter.IsEmpty()):
            if self.catalog == 2: 
                information = False
            elif (self.maxLabel_enter.IsEmpty() | self.minLabel_enter.IsEmpty()):
                information = False
        
        if information:

            label = int(self.label_choice.GetCurrentSelection())
            normalize = int(self.normalize_choice.GetCurrentSelection())
            pca = int(self.pca_choice.GetCurrentSelection())

            if self.catalog != 2:
                max_label = int(self.maxLabel_enter.GetValue())
                min_label = int(self.minLabel_enter.GetValue())

            if self.split_enter.IsEmpty():
                character = ''
            else:
                character = self.split_enter.GetValue()
            master = self.master_enter.GetValue()
            if self.catalog == 0:
                showpic, string = SVMModel(self.dataPath, label, max_label, min_label, character, master, normalize, pca)
            elif self.catalog == 1:
                showpic, string = LogisticRegressionModel(self.dataPath, label, max_label, min_label, character, master, normalize, pca)
            elif self.catalog == 2:
                string = NaiveBayesModel(self.dataPath, label, character, master, normalize, pca)
        
        else:
            string = "The information is not complete.\nPlease check again."
        
        self.result.SetValue(string)

        if showpic == 1:
            frame = showResult(parent = None, id = 2, catalog = self.catalog)
            frame.Show()

        
    def selectData(self, event):
        id = self.id       
        panel = self.panel
        clear = " " * 100
        dig = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.*", wx.OPEN)

        if dig.ShowModal() == wx.ID_OK:
            filename = dig.GetFilename()
            path = dig.GetDirectory()
            self.dataPath = path + '/' + filename
            self.textName.SetLabel(filename)
            #Clean the information
            self.maxLabel_enter.SetValue("")
            self.minLabel_enter.SetValue("")
            self.split_enter.SetValue("")
            self.master_enter.SetValue("")        
            self.result.SetValue("")

class showResult(wx.Frame):


    def __init__(self, parent, id, catalog):

        if catalog == 0:
            wx.Frame.__init__(self, parent, id, "SVM Result", size = (500, 400))
        elif catalog == 1:
            wx.Frame.__init__(self, parent, id, "Logistic Regression Result", size = (500, 400))
        
       
        panel = wx.Panel(self)

        bitmap = wx.Bitmap('result.jpg')
        image = wx.ImageFromBitmap(bitmap)
        image = image.Scale(500, 400, wx.IMAGE_QUALITY_HIGH)
        bitmap = wx.BitmapFromImage(image)

        control = wx.StaticBitmap(self, -1, bitmap)
        control.SetPosition((1, 1))
    
    
    

class regression(wx.Frame):
        
    def __init__(self, parent, id, catalog):

        self.catalog = catalog
        if self.catalog == 0:
            wx.Frame.__init__(self, parent, id, "Lasso Regression", size = (600, 700))
        elif self.catalog == 1:
            wx.Frame.__init__(self, parent, id, "Linear Regression", size = (600, 700))
        elif self.catalog == 2:
            wx.Frame.__init__(self, parent, id, "Ridge Regression", size = (600, 700))

        self.panel = wx.Panel(self) 
        self.id = id 
        self.dataPath = ""
        self.readFirst = True        
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox_select = wx.BoxSizer(wx.HORIZONTAL)                        
        hbox_label = wx.BoxSizer(wx.HORIZONTAL)
        hbox_split = wx.BoxSizer(wx.HORIZONTAL) 
        hbox_normalize = wx.BoxSizer(wx.HORIZONTAL)
        hbox_pca = wx.BoxSizer(wx.HORIZONTAL)
        hbox_master = wx.BoxSizer(wx.HORIZONTAL)           
        hbox_start = wx.BoxSizer(wx.HORIZONTAL)            
        hbox_result = wx.BoxSizer(wx.HORIZONTAL)
    
        #Data Selection
        selectButton = wx.Button(self.panel, label = "Data")
        font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        selectButton.SetFont(font)
        self.textName = wx.StaticText(self.panel, id, "", style = wx.ALIGN_LEFT)        
        hbox_select.AddSpacer(20)
        hbox_select.Add(selectButton, 1)
        hbox_select.Add(self.textName, 3)
           
        
        #Location
        label_text = wx.StaticText(self.panel, id, "Label(location): ", style = wx.ALIGN_LEFT)
        self.label_choice = wx.Choice(self.panel, id, (-1, -1), (-1, -1), ['First', 'End'])        
        hbox_label.AddSpacer(20)
        hbox_label.Add(label_text, 1)
        hbox_label.Add(self.label_choice, 1) 

        #Normalization
        normalize_text =  wx.StaticText(self.panel, id, "Normalization (min-max): ", style = wx.ALIGN_LEFT)
        self.normalize_choice = wx.Choice(self.panel, id, (-1, -1), (-1, -1), ['No', 'Yes'])        
        hbox_normalize.AddSpacer(20)
        hbox_normalize.Add(normalize_text, 2)
        hbox_normalize.Add(self.normalize_choice, 1)

        #PCA
        pca_text =  wx.StaticText(self.panel, id, "PCA: ", style = wx.ALIGN_LEFT)
        self.pca_choice = wx.Choice(self.panel, id, (-1, -1), (-1, -1), ['No', 'Yes'])        
        hbox_pca.AddSpacer(20)
        hbox_pca.Add(pca_text, 1)
        hbox_pca.Add(self.pca_choice, 1)

      
        #Split Character
        split_text = wx.StaticText(self.panel, id, "Split Character: ", style = wx.ALIGN_LEFT)
        self.split_enter = wx.TextCtrl(self.panel, id, size = (20, -1), style = wx.TE_PROCESS_TAB)
        hbox_split.AddSpacer(20)
        hbox_split.Add(split_text, 1)
        hbox_split.Add(self.split_enter)
        
        #Master
        master_text = wx.StaticText(self.panel, id, "Master: ", style = wx.ALIGN_LEFT)
        self.master_enter = wx.TextCtrl(self.panel, id, size = (50, -1), style = wx.TE_PROCESS_TAB)
        hbox_master.AddSpacer(20)
        hbox_master.Add(master_text, 1)
        hbox_master.Add(self.master_enter)

        checkButton = wx.Button(self.panel, label = "Start")
        font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        
        checkButton.SetFont(font)        
        hbox_start.AddSpacer(20)
        hbox_start.Add(checkButton, 1)
        
        #result 
        self.result = wx.TextCtrl(self.panel, id, style = wx.TE_READONLY | wx.TE_MULTILINE)       
        hbox_result.Add(self.result, 1, wx.EXPAND)
        
        # Default path for openning data
        self.dirname = '/Users/Terry/Work/spark-1.0.1-bin-hadoop1/project'
        
        # Box setting
        vbox.AddSpacer(10)
        vbox.Add(hbox_select, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_label, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_normalize, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_pca, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_split, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_master, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_start, flag=wx.LEFT|wx.RIGHT|wx.TOP)
        vbox.Add((-1, 10))
        vbox.Add(hbox_result, 1, flag = wx.EXPAND | wx.LEFT | wx.RIGHT, border = 20)
        vbox.Add((-1, 20))
        self.panel.SetSizer(vbox)
        

        # button event
        self.Bind(wx.EVT_BUTTON,self.selectData, selectButton)
        self.Bind(wx.EVT_BUTTON, self.start, checkButton)

    def start(self, event):
        id = self.id       
        panel = self.panel
        information = True        

        if (self.master_enter.IsEmpty()):
            information = False
        
        if information:

            label = int(self.label_choice.GetCurrentSelection())
            master = self.master_enter.GetValue()
            normalize = int(self.normalize_choice.GetCurrentSelection())
            pca = int(self.pca_choice.GetCurrentSelection())
           
            if self.split_enter.IsEmpty():
                character = ' '
            else:
                character = self.split_enter.GetValue()
            

            if self.catalog == 0:
                string = LassoModel(self.dataPath, label, normalize, character, master, pca)
            elif self.catalog == 1:
                string = LinearRegressionModel(self.dataPath, label, normalize, character, master, pca)
            elif self.catalog == 2:
                string = RidgeRegressionModel(self.dataPath, label, normalize, character, master, pca)
        
        else:
            string = "The information is not complete.\nPlease check again."
        self.result.SetValue(string)

    def selectData(self, event):
        id = self.id       
        panel = self.panel
        clear = " " * 100
        dig = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.*", wx.OPEN)

        if dig.ShowModal() == wx.ID_OK:
            filename = dig.GetFilename()
            path = dig.GetDirectory()
            self.dataPath = path + '/' + filename
            self.textName.SetLabel(filename)
            #Clean the information
            self.split_enter.SetValue("")
            self.master_enter.SetValue("")        
            self.result.SetValue("")

if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = bucky(parent = None, id = -1)
    frame.Show()
    app.MainLoop()

