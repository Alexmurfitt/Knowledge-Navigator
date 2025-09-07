import React from 'react';
import { X, Clock, Target, Globe, FileText } from 'lucide-react';

interface InfoPopupProps {
  isOpen: boolean;
  onClose: () => void;
  info: any;
}

const InfoPopup: React.FC<InfoPopupProps> = ({ isOpen, onClose, info }) => {
  if (!isOpen || !info) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-start">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity duration-300"
        onClick={onClose}
      />
      
      {/* Popup */}
      <div className="relative bg-white rounded-r-2xl shadow-2xl w-full max-w-md h-full overflow-y-auto transform transition-transform duration-300 slide-in-left">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">Response Details</h2>
            <button
              onClick={onClose}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-all duration-200"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="space-y-6">
            {/* Confidence Score */}
            {info.confidence && (
              <div className="bg-blue-50 p-4 rounded-xl">
                <div className="flex items-center space-x-2 mb-2">
                  <Target className="w-4 h-4 text-blue-600" />
                  <span className="font-medium text-blue-900">Confidence Score</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex-1 bg-blue-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${(info.confidence * 100)}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold text-blue-900">
                    {Math.round(info.confidence * 100)}%
                  </span>
                </div>
              </div>
            )}

            {/* Processing Time */}
            {info.processingTime && (
              <div className="bg-green-50 p-4 rounded-xl">
                <div className="flex items-center space-x-2 mb-1">
                  <Clock className="w-4 h-4 text-green-600" />
                  <span className="font-medium text-green-900">Processing Time</span>
                </div>
                <span className="text-sm text-green-700">{info.processingTime}</span>
              </div>
            )}

            {/* Internet Usage */}
            {info.internetUsed !== undefined && (
              <div className={`p-4 rounded-xl ${info.internetUsed ? 'bg-purple-50' : 'bg-gray-50'}`}>
                <div className="flex items-center space-x-2 mb-1">
                  <Globe className={`w-4 h-4 ${info.internetUsed ? 'text-purple-600' : 'text-gray-500'}`} />
                  <span className={`font-medium ${info.internetUsed ? 'text-purple-900' : 'text-gray-700'}`}>
                    Internet Search
                  </span>
                </div>
                <span className={`text-sm ${info.internetUsed ? 'text-purple-700' : 'text-gray-600'}`}>
                  {info.internetUsed ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            )}

            {/* Sources */}
            {info.sources && info.sources.length > 0 && (
              <div className="bg-orange-50 p-4 rounded-xl">
                <div className="flex items-center space-x-2 mb-3">
                  <FileText className="w-4 h-4 text-orange-600" />
                  <span className="font-medium text-orange-900">Source Documents</span>
                </div>
                <div className="space-y-2">
                  {info.sources.map((source: string, index: number) => (
                    <div key={index} className="bg-white p-2 rounded-lg border border-orange-200">
                      <span className="text-sm text-orange-800">{source}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default InfoPopup;