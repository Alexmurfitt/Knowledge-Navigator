import React from 'react';
import { FileText, Trash2, Eye, Clock, CheckCircle } from 'lucide-react';
import { Document } from '../utils/types';

interface DocumentCardProps {
  document: Document;
  onDelete: () => void;
  onView: () => void;
}

const DocumentCard: React.FC<DocumentCardProps> = ({ document, onDelete, onView }) => {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'uploaded':
        return {
          icon: CheckCircle,
          color: 'text-green-600',
          bg: 'bg-green-100',
          label: 'Uploaded'
        };
      case 'processing':
        return {
          icon: Clock,
          color: 'text-orange-600',
          bg: 'bg-orange-100',
          label: 'Processing'
        };
      default:
        return {
          icon: Clock,
          color: 'text-gray-600',
          bg: 'bg-gray-100',
          label: 'Pending'
        };
    }
  };

  const statusConfig = getStatusConfig(document.status);
  const StatusIcon = statusConfig.icon;

  return (
    <div className="group bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-all duration-300 hover:-translate-y-1 document-card-gradient">
      {/* Document Icon */}
      <div className="flex items-center justify-center w-16 h-16 bg-blue-100 rounded-xl mb-4 mx-auto">
        <FileText className="w-8 h-8 text-blue-600" />
      </div>

      {/* Document Name */}
      <h3 className="text-lg font-semibold text-gray-900 mb-2 text-center group-hover:text-blue-600 transition-colors duration-200">
        {document.name}
      </h3>

      {/* Status */}
      <div className="flex items-center justify-center space-x-2 mb-4">
        <div className={`p-1 rounded-full ${statusConfig.bg}`}>
          <StatusIcon className={`w-4 h-4 ${statusConfig.color}`} />
        </div>
        <span className={`text-sm font-medium ${statusConfig.color}`}>
          {statusConfig.label}
        </span>
      </div>

      {/* Upload Date */}
      <p className="text-xs text-gray-500 text-center mb-4">
        Uploaded {document.uploadedAt.toLocaleDateString()}
      </p>

      {/* Action Buttons */}
      <div className="flex items-center justify-center space-x-2">
        <button
          onClick={onView}
          className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-all duration-200 hover:scale-110"
          title="View document"
        >
          <Eye className="w-4 h-4" />
        </button>
        <button
          onClick={onDelete}
          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-all duration-200 hover:scale-110"
          title="Delete document"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default DocumentCard;