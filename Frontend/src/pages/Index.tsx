import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Leaf, Shield, Search, ArrowRight } from "lucide-react";

const plants = [
  { id: "tomato", name: "Tomato", description: "Common vegetable crop susceptible to various fungal and bacterial diseases" },
  { id: "potato", name: "Potato", description: "Important staple crop affected by blight and other diseases" },
  { id: "bean", name: "Bean", description: "Legume crop with various leaf spot and blight diseases" },
  { id: "grape", name: "Grape", description: "Vine fruit susceptible to fungal diseases like black rot and powdery mildew" }
];

const Index = () => {
  const [selectedPlant, setSelectedPlant] = useState("");
  const navigate = useNavigate();

  const handlePlantSelection = () => {
    if (selectedPlant) {
      navigate(`/plant/${selectedPlant}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-green-100 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Leaf className="h-8 w-8 text-green-600" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                PlantCare AI
              </h1>
            </div>
            <Button variant="outline" className="border-green-200 text-green-700 hover:bg-green-50">
              Learn More
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-green-700 via-emerald-600 to-teal-600 bg-clip-text text-transparent">
            Detect Plant Diseases with AI
          </h2>
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Protect your crops and garden with our advanced AI-powered plant disease detection system. 
            Get instant diagnosis and treatment recommendations to keep your plants healthy.
          </p>
          
          <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-green-100 mb-12">
            <h3 className="text-2xl font-semibold mb-6 text-gray-800">Select Your Plant</h3>
            <div className="flex flex-col sm:flex-row gap-4 items-center justify-center max-w-md mx-auto">
              <Select onValueChange={setSelectedPlant}>
                <SelectTrigger className="w-full border-green-200 focus:ring-green-500">
                  <SelectValue placeholder="Choose a plant type" />
                </SelectTrigger>
                <SelectContent className="bg-white border-green-100">
                  {plants.map((plant) => (
                    <SelectItem key={plant.id} value={plant.id} className="hover:bg-green-50">
                      {plant.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button 
                onClick={handlePlantSelection}
                disabled={!selectedPlant}
                className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 w-full sm:w-auto"
              >
                Analyze Plant
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-16">
        <h3 className="text-3xl font-bold text-center mb-12 text-gray-800">How We Solve Plant Disease Problems</h3>
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="bg-white/70 backdrop-blur-sm border-green-100 hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mb-4">
                <Search className="h-6 w-6 text-white" />
              </div>
              <CardTitle className="text-green-800">AI-Powered Detection</CardTitle>
              <CardDescription>
                Advanced machine learning algorithms analyze plant images to identify diseases with high accuracy
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Our neural networks are trained on thousands of plant images to recognize patterns and symptoms
                of various diseases across different plant species.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-white/70 backdrop-blur-sm border-green-100 hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-green-600 rounded-lg flex items-center justify-center mb-4">
                <Leaf className="h-6 w-6 text-white" />
              </div>
              <CardTitle className="text-green-800">Plant-Specific Analysis</CardTitle>
              <CardDescription>
                Specialized models for different plant types ensure accurate diagnosis for your specific crops
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Each plant species has unique disease patterns. Our specialized models understand these
                differences to provide more accurate diagnoses.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-white/70 backdrop-blur-sm border-green-100 hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg flex items-center justify-center mb-4">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <CardTitle className="text-green-800">Treatment Recommendations</CardTitle>
              <CardDescription>
                Get actionable treatment advice and prevention strategies for identified diseases
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Beyond detection, we provide comprehensive treatment plans including organic and chemical
                solutions to help restore your plants to health.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Problem Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-2xl p-8 border border-red-100">
          <h3 className="text-3xl font-bold text-center mb-8 text-gray-800">The Plant Disease Challenge</h3>
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h4 className="text-xl font-semibold mb-4 text-red-700">Global Impact</h4>
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start">
                  <span className="text-red-500 mr-2">•</span>
                  Plant diseases cause 20-40% crop yield losses globally
                </li>
                <li className="flex items-start">
                  <span className="text-red-500 mr-2">•</span>
                  Early detection can save up to 80% of affected crops
                </li>
                <li className="flex items-start">
                  <span className="text-red-500 mr-2">•</span>
                  Traditional diagnosis requires expert knowledge and time
                </li>
                <li className="flex items-start">
                  <span className="text-red-500 mr-2">•</span>
                  Misdiagnosis leads to ineffective treatments and further losses
                </li>
              </ul>
            </div>
            <div>
              <h4 className="text-xl font-semibold mb-4 text-green-700">Our Solution</h4>
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">•</span>
                  Instant AI-powered diagnosis from smartphone photos
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">•</span>
                  90%+ accuracy across multiple plant diseases
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">•</span>
                  Accessible to farmers and gardeners worldwide
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">•</span>
                  Real-time treatment recommendations and prevention tips
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-sm border-t border-green-100 mt-16">
        <div className="container mx-auto px-4 py-8 text-center">
          <p className="text-gray-600">© 2024 PlantCare AI. Protecting plants with artificial intelligence.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
