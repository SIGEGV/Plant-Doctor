import { useState ,useRef} from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { ArrowLeft, Upload, Camera, AlertTriangle, CheckCircle, Leaf, X } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const plantData = {
  tomato: {
  name: "Tomato",
  diseases: [
    {
      name: "Bacterial Spot",
      symptoms: "Small dark spots on leaves and fruit, yellowing around spots",
      causes: "Xanthomonas bacteria, spread by water splash and insects",
      treatment: "Copper sprays, resistant varieties, avoid overhead watering"
    },
    {
      name: "Early Blight",
      symptoms: "Dark spots with concentric rings on older leaves, yellowing and wilting",
      causes: "Alternaria solani fungus, favored by warm humid conditions",
      treatment: "Fungicide application, proper spacing, crop rotation"
    },
    {
      name: "Healthy",
      symptoms: "No disease symptoms, healthy plant",
      causes: "N/A",
      treatment: "N/A"
    },
    {
      name: "Late Blight",
      symptoms: "Water-soaked lesions on leaves, white fuzzy growth on undersides",
      causes: "Phytophthora infestans, thrives in cool wet weather",
      treatment: "Copper-based fungicides, remove affected plants, improve air circulation"
    },
    {
      name: "Leaf Mold",
      symptoms: "Yellow spots on upper leaf surface, mold growth on undersides",
      causes: "Fulvia fulva fungus, humid conditions",
      treatment: "Fungicides, improve air circulation, avoid wetting leaves"
    },
    {
      name: "Septoria Leaf Spot",
      symptoms: "Small circular spots with dark borders on leaves",
      causes: "Septoria lycopersici fungus, spreads in wet weather",
      treatment: "Fungicides, remove infected leaves, crop rotation"
    },
    {
      name: "Spider Mites",
      symptoms: "Tiny spots on leaves, webbing under leaves, leaf discoloration",
      causes: "Tetranychus mites, dry and dusty conditions",
      treatment: "Miticides, insecticidal soap, maintain humidity"
    },
    {
      name: "Target Spot",
      symptoms: "Brown spots with concentric rings on leaves and stems",
      causes: "Corynespora cassiicola fungus",
      treatment: "Fungicides, remove infected debris"
    },
    {
      name: "Tomato Mosaic Virus",
      symptoms: "Mottled yellow-green pattern on leaves, stunted growth",
      causes: "Tobacco mosaic virus, spread by insects and contaminated tools",
      treatment: "Remove infected plants, control aphids, use virus-free seeds"
    },
    {
      name: "Tomato Yellow Leaf Curl Virus",
      symptoms: "Yellowing and curling of leaves, stunted plant growth",
      causes: "Begomovirus transmitted by whiteflies",
      treatment: "Control whiteflies, remove infected plants, resistant varieties"
    }
  ]
},
  potato: {
    name: "Potato",
    diseases: [
      {
        name: "Late Blight",
        symptoms: "Water-soaked lesions, white fungal growth, tuber rot",
        causes: "Phytophthora infestans, cool moist conditions",
        treatment: "Fungicide sprays, proper storage, resistant varieties"
      },
      {
        name: "Early Blight",
        symptoms: "Target spot lesions on leaves, premature defoliation",
        causes: "Alternaria solani, stressed plants more susceptible",
        treatment: "Fungicide application, proper nutrition, crop rotation"
      },
      {
        name: "Scab",
        symptoms: "Rough, corky lesions on tuber surface",
        causes: "Streptomyces bacteria in alkaline soils",
        treatment: "Lower soil pH, avoid fresh manure, resistant varieties"
      }
    ]
  },
  bean: {
    name: "Bean",
    diseases: [
      {
        name: "Angular Leaf Spot",
        symptoms: "Small angular water-soaked spots on leaves, yellowing and defoliation",
        causes: "Pseudomonas syringae bacteria, spread by water splash and insects",
        treatment: "Copper-based bactericides, crop rotation, avoid overhead irrigation"
      },
      {
        name: "Bean Rust",
        symptoms: "Small reddish-brown pustules on leaf undersides, yellowing leaves",
        causes: "Uromyces appendiculatus fungus, favored by warm humid conditions",
        treatment: "Fungicide sprays, resistant varieties, proper plant spacing"
      },
      {
        name: "Healthy",
        symptoms: "No visible disease symptoms, normal green foliage",
        causes: "No disease present, good growing conditions",
        treatment: "Continue good cultural practices, regular monitoring"
      }
    ]
  },
  grape: {
    name: "Grape",
  diseases: [
    {
      name: "Black Rot",
      symptoms: "Small brown spots on leaves that enlarge and turn black, shriveled black fruit (mummies)",
      causes: "Guignardia bidwellii fungus, spread by rain and humid conditions",
      treatment: "Prune infected areas, apply fungicides, ensure good air circulation"
    },
    {
      name: "ESCA",
      symptoms: "Interveinal chlorosis, leaf scorching, black streaks in wood",
      causes: "Fungal pathogens including Phaeoacremonium spp. and Phaeomoniella chlamydospora",
      treatment: "Remove infected vines, avoid pruning wounds during wet conditions"
    },
    {
      name: "Leaf Blight",
      symptoms: "Irregular brown spots on leaves, leaf drop, reduced vigor",
      causes: "Caused by fungal pathogens, often linked to poor ventilation and high humidity",
      treatment: "Remove infected leaves, apply appropriate fungicide, improve airflow"
    },
    {
      name: "Healthy",
      symptoms: "No visible spots, uniform green leaves, vigorous growth",
      causes: "Optimal conditions and disease-free environment",
      treatment: "Maintain proper vineyard management and regular monitoring"
    }
  ]
}
};

const PlantPage = () => {
  const { plantId } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showResultDialog, setShowResultDialog] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const plant = plantData[plantId as keyof typeof plantData];

  if (!plant) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Plant not found</h1>
          <Button onClick={() => navigate("/")} variant="outline">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
        </div>
      </div>
    );
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPredictionResult(null);
      
      // Create image preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select an image first",
        variant: "destructive"
      });
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`http://127.0.0.1:5000/api/${plantId}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setPredictionResult(data);
      setShowResultDialog(true);
      
      toast({
        title: "Analysis Complete",
        description: "Your plant image has been analyzed successfully",
      });
    } catch (error) {
      console.error('Upload error:', error);
      toast({
        title: "Upload Failed",
        description: "There was an error analyzing your image. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsUploading(false);
    }
  };

  const getDiseaseInfo = (diseaseName: string) => {
    return plant.diseases.find(disease => 
      disease.name.toLowerCase().includes(diseaseName.toLowerCase()) ||
      diseaseName.toLowerCase().includes(disease.name.toLowerCase())
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-green-100">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Button 
              onClick={() => navigate("/")} 
              variant="ghost" 
              className="text-green-700 hover:bg-green-50"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Button>
            <div className="flex items-center space-x-2">
              <Leaf className="h-6 w-6 text-green-600" />
              <span className="font-semibold text-green-800">{plant.name} Disease Detection</span>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Plant Info Section */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-700 to-emerald-600 bg-clip-text text-transparent">
            {plant.name} Disease Analysis
          </h1>
          <p className="text-lg text-gray-600">
            Upload an image of your {plant.name.toLowerCase()} plant to detect diseases and get treatment recommendations.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section with Preview */}
          <Card className="bg-white/70 backdrop-blur-sm border-green-100">
            <CardHeader>
              <CardTitle className="flex items-center text-green-800">
                <Camera className="mr-2 h-5 w-5" />
                Upload Plant Image
              </CardTitle>
              <CardDescription>
                Take a clear photo of the affected plant parts for accurate analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {imagePreview ? (
                <div className="relative">
                  <img 
                    src={imagePreview} 
                    alt="Plant preview" 
                    className="w-full h-64 object-contain rounded-lg border border-green-200"
                  />
                  <button
                    onClick={handleRemoveImage}
                    className="absolute top-2 right-2 bg-white/80 rounded-full p-1 hover:bg-white transition-colors"
                  >
                    <X className="h-4 w-4 text-gray-600" />
                  </button>
                </div>
              ) : (
                <div className="border-2 border-dashed border-green-200 rounded-lg p-8 text-center hover:border-green-300 transition-colors">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                    ref={fileInputRef}
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <Upload className="mx-auto h-12 w-12 text-green-500 mb-4" />
                    <p className="text-lg font-medium text-gray-700 mb-2">
                      Click to upload an image
                    </p>
                    <p className="text-sm text-gray-500">
                      Supports JPEG, PNG, and other image formats
                    </p>
                  </label>
                </div>
              )}
              
              {selectedFile && (
                <Alert className="border-green-200 bg-green-50">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-700">
                    Selected: {selectedFile.name}
                  </AlertDescription>
                </Alert>
              )}

              <Button 
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
                className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
              >
                {isUploading ? "Analyzing..." : "Analyze Image"}
                <Upload className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>

          {/* Disease Information */}
          <Card className="bg-white/70 backdrop-blur-sm border-green-100">
            <CardHeader>
              <CardTitle className="text-green-800">Common {plant.name} Diseases</CardTitle>
              <CardDescription>
                Learn about diseases that commonly affect {plant.name.toLowerCase()} plants
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {plant.diseases.map((disease, index) => (
                  <div key={index} className="border-l-4 border-green-400 pl-4 py-2">
                    <h4 className="font-semibold text-gray-800 mb-2">{disease.name}</h4>
                    <div className="space-y-2 text-sm">
                      <div>
                        <span className="font-medium text-red-600">Symptoms:</span>
                        <span className="text-gray-600 ml-2">{disease.symptoms}</span>
                      </div>
                      <div>
                        <span className="font-medium text-orange-600">Causes:</span>
                        <span className="text-gray-600 ml-2">{disease.causes}</span>
                      </div>
                      <div>
                        <span className="font-medium text-green-600">Treatment:</span>
                        <span className="text-gray-600 ml-2">{disease.treatment}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tips Section */}
        <Card className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-100">
          <CardHeader>
            <CardTitle className="text-blue-800">Photography Tips for Best Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">•</span>
                  Take photos in good natural lighting
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">•</span>
                  Focus on affected leaves or plant parts
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">•</span>
                  Ensure the image is clear and not blurry
                </li>
              </ul>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">•</span>
                  Include both healthy and diseased areas
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">•</span>
                  Avoid shadows and reflections
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">•</span>
                  Take multiple angles if symptoms are unclear
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Prediction Result Dialog with Image Preview */}
      <Dialog open={showResultDialog} onOpenChange={setShowResultDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center text-green-800">
              <Leaf className="mr-2 h-5 w-5" />
              Disease Detection Results
            </DialogTitle>
            <DialogDescription>
              Analysis results for your {plant.name.toLowerCase()} plant
            </DialogDescription>
          </DialogHeader>
          
          {predictionResult && (
            <div className="space-y-4">
              <div className="flex flex-col md:flex-row gap-6">
                {imagePreview && (
                  <div className="w-full md:w-1/3">
                    <div className="border border-gray-200 rounded-lg overflow-hidden">
                      <img 
                        src={imagePreview} 
                        alt="Analyzed plant" 
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                )}
                <div className="w-full md:w-2/3 space-y-4">
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
                    <h3 className="font-semibold text-blue-800 mb-2">Prediction Result</h3>
                    <p className="text-blue-700">
                      <strong>Detected Disease:</strong> {predictionResult.prediction || predictionResult.class || 'Unknown'}
                    </p>
                    {predictionResult.confidence && (
                      <p className="text-blue-700 mt-1">
                        <strong>Confidence:</strong> {(predictionResult.confidence * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>

                  {(() => {
                    const diseaseInfo = getDiseaseInfo(predictionResult.prediction || predictionResult.class || '');
                    if (diseaseInfo) {
                      return (
                        <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
                          <h3 className="font-semibold text-green-800 mb-3">Treatment Information</h3>
                          <div className="space-y-3">
                            <div>
                              <span className="font-medium text-red-600">Symptoms:</span>
                              <p className="text-gray-700 mt-1">{diseaseInfo.symptoms}</p>
                            </div>
                            <div>
                              <span className="font-medium text-orange-600">Causes:</span>
                              <p className="text-gray-700 mt-1">{diseaseInfo.causes}</p>
                            </div>
                            <div>
                              <span className="font-medium text-green-600">Treatment:</span>
                              <p className="text-gray-700 mt-1">{diseaseInfo.treatment}</p>
                            </div>
                          </div>
                        </div>
                      );
                    }
                    return (
                      <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                        <p className="text-yellow-800">
                          No specific treatment information available for this detection. 
                          Please consult with a plant disease specialist for proper treatment advice.
                        </p>
                      </div>
                    );
                  })()}
                </div>
              </div>

              <div className="flex justify-end space-x-2 pt-4">
                <Button 
                  variant="outline" 
                  onClick={() => setShowResultDialog(false)}
                >
                  Close
                </Button>
                <Button 
                  onClick={() => {
                    setShowResultDialog(false);
                    setSelectedFile(null);
                    setImagePreview(null);
                  }}
                  className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                >
                  Analyze Another Image
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PlantPage;