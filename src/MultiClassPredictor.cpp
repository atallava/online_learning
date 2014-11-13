#include <iomanip>
#include <ol/MultiClassPredictor.h>

using namespace ol;

MultiClassPredictor::MultiClassPredictor() : stream_rounds_(0),
					     stream_label_count_(NUM_CLASSES, 0),
					     stream_confusion_matrix_(NUM_CLASSES, 
								     std::vector<double>(NUM_CLASSES,0))						       
{} 

void MultiClassPredictor::updateStreamLogs(Label true_label, Label predicted_label)
{
    stream_rounds_ += 1;
    stream_label_count_[true_label] += 1;
    stream_confusion_matrix_[true_label][predicted_label] += 1;
}

void MultiClassPredictor::printStreamLogs()
{
    std::cout << "Stream logs: \n\n";
    std::cout << "Number of examples seen: " << stream_rounds_ << "\n\n";
    std::cout << std::left << std::setw(20) << "CLASS NAME" 
	      << std::left << std::setw(20) << "CLASS FREQUENCY" 
	      << std::left << std::setw(20) << "PER CLASS ACCURACY" << std::endl;

    double accuracy = 0;
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
	std::cout << std::left << std::setw(20) << CLASS_NAMES[i] 
		  << std::left << std::setw(20) << stream_label_count_[i]/stream_rounds_ 
		  << std::left << std::setw(20) << stream_confusion_matrix_[i][i]/stream_label_count_[i] 
		  << std::endl;
	accuracy += static_cast<double>(stream_confusion_matrix_[i][i]);
    }
    accuracy /= stream_rounds_;

    std::cout << "\n";
    std::cout << "Confusion Matrix: \n";
    for (size_t i = 0; i < stream_confusion_matrix_.size(); ++i) {
	for (size_t j = 0; j < stream_confusion_matrix_[0].size(); ++j) {
	    std::cout << std::left << std::setw(10) << stream_confusion_matrix_[i][j];
	}
	std::cout << "\n\n";
    }

    std::cout << "Overall accuracy: " << accuracy << "\n\n";
}


