import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * Training Chart Component
 * 
 * Provides real-time visualization of training progress, replicating
 * the matplotlib charts from the Python GUI. Shows episode rewards
 * and lengths over time with smooth animations.
 */
const TrainingChart = ({ data }) => {
  // Transform data for recharts format
  const chartData = data.episodes.map((episode, index) => ({
    episode,
    reward: data.rewards[index] || 0,
    length: data.lengths[index] || 0,
    // Calculate moving average for smoother visualization
    avgReward: index >= 9 ? 
      data.rewards.slice(Math.max(0, index - 9), index + 1)
        .reduce((sum, r) => sum + r, 0) / Math.min(10, index + 1) : 
      data.rewards[index] || 0
  }));

  // Custom tooltip for better data display
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="font-semibold">{`Episode: ${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.value.toFixed(2)}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">
            No Training Data Yet
          </h3>
          <p className="text-gray-500">
            Start training to see real-time progress visualization
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Episode Rewards Chart */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-gray-800">
          📈 Episode Rewards
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="episode" 
              label={{ value: 'Episode', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              label={{ value: 'Reward', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="reward" 
              stroke="#3B82F6" 
              strokeWidth={1}
              dot={false}
              name="Episode Reward"
              opacity={0.7}
            />
            <Line 
              type="monotone" 
              dataKey="avgReward" 
              stroke="#EF4444" 
              strokeWidth={2}
              dot={false}
              name="Moving Average (10 episodes)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Episode Lengths Chart */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-gray-800">
          ⏱️ Episode Lengths
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="episode" 
              label={{ value: 'Episode', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              label={{ value: 'Steps', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="length" 
              stroke="#10B981" 
              strokeWidth={2}
              dot={false}
              name="Episode Length"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Training Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">
            {chartData.length}
          </div>
          <div className="text-sm text-blue-800">Episodes Completed</div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-green-600">
            {chartData.length > 0 ? 
              chartData[chartData.length - 1].reward.toFixed(1) : '0.0'}
          </div>
          <div className="text-sm text-green-800">Latest Reward</div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-purple-600">
            {chartData.length > 0 ? 
              (data.rewards.reduce((sum, r) => sum + r, 0) / data.rewards.length).toFixed(1) : '0.0'}
          </div>
          <div className="text-sm text-purple-800">Average Reward</div>
        </div>
        
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-orange-600">
            {chartData.length > 0 ? Math.max(...data.rewards).toFixed(1) : '0.0'}
          </div>
          <div className="text-sm text-orange-800">Best Reward</div>
        </div>
      </div>
    </div>
  );
};

export default TrainingChart;
