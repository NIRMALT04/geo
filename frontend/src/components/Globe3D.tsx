import React, { useRef, useState, useEffect, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Html } from '@react-three/drei'
import { Vector3, Mesh, BufferGeometry, LineBasicMaterial, BufferAttribute } from 'three'
import * as THREE from 'three'
import { motion } from 'framer-motion'
import { useEODataStore } from '../stores/eoDataStore'
import { EOTile, AnalysisResult } from '../types/eo'

interface Globe3DProps {
  onTileClick?: (tile: EOTile) => void
  selectedTile?: EOTile | null
  analysisResults?: AnalysisResult[]
}

const EARTH_RADIUS = 5

// Convert lat/lng to 3D coordinates
const latLngToVector3 = (lat: number, lng: number, radius: number = EARTH_RADIUS): Vector3 => {
  const phi = (90 - lat) * (Math.PI / 180)
  const theta = (lng + 180) * (Math.PI / 180)
  
  return new Vector3(
    -(radius * Math.sin(phi) * Math.cos(theta)),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  )
}

// Enhanced Earth sphere with gradient material
const EarthSphere: React.FC = () => {
  const meshRef = useRef<Mesh>(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.01
    }
  })

  return (
    <group>
      {/* Main Earth sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
        <meshLambertMaterial 
          color="#2563eb"
          transparent
          opacity={0.9}
        />
      </mesh>
      
      {/* Atmosphere glow */}
      <mesh>
        <sphereGeometry args={[EARTH_RADIUS * 1.02, 32, 32]} />
        <meshBasicMaterial 
          color="#60a5fa"
          transparent
          opacity={0.1}
          side={THREE.BackSide}
        />
      </mesh>
      
      {/* Wireframe overlay for continents */}
      <mesh>
        <sphereGeometry args={[EARTH_RADIUS * 1.001, 32, 32]} />
        <meshBasicMaterial 
          color="#93c5fd"
          wireframe
          transparent
          opacity={0.3}
        />
      </mesh>
    </group>
  )
}

// EO Tile visualization
const EOTileMarker: React.FC<{ tile: EOTile; onClick: () => void; isSelected: boolean }> = ({
  tile,
  onClick,
  isSelected
}) => {
  const meshRef = useRef<Mesh>(null)
  const position = useMemo(() => 
    latLngToVector3(tile.latitude, tile.longitude, EARTH_RADIUS + 0.1), 
    [tile.latitude, tile.longitude]
  )

  useFrame((state) => {
    if (meshRef.current && isSelected) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.1)
    }
  })

  return (
    <mesh
      ref={meshRef}
      position={position}
      onClick={onClick}
      scale={isSelected ? 1.2 : 1}
    >
      <sphereGeometry args={[0.05, 16, 16]} />
      <meshBasicMaterial 
        color={isSelected ? '#ff6b6b' : tile.analysisStatus === 'completed' ? '#51cf66' : '#ffd43b'} 
        transparent
        opacity={0.8}
      />
      {tile.analysisStatus === 'processing' && (
        <Html>
          <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />
        </Html>
      )}
    </mesh>
  )
}

// Analysis result visualization overlay
const AnalysisOverlay: React.FC<{ result: AnalysisResult }> = ({ result }) => {
  const position = useMemo(() => 
    latLngToVector3(result.latitude, result.longitude, EARTH_RADIUS + 0.2), 
    [result.latitude, result.longitude]
  )

  return (
    <Html position={position}>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-black/80 text-white p-3 rounded-lg max-w-xs backdrop-blur-sm"
      >
        <h4 className="font-semibold text-sm mb-2">{result.classification}</h4>
        <p className="text-xs text-gray-300 mb-2">{result.description}</p>
        <div className="flex justify-between text-xs">
          <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
          <span>{new Date(result.timestamp).toLocaleDateString()}</span>
        </div>
      </motion.div>
    </Html>
  )
}

// GeoJSON boundary visualization
const GeoJSONBoundaries: React.FC<{ geoJsonData: any }> = ({ geoJsonData }) => {
  const linesRef = useRef<THREE.Group>(null)

  const boundaryLines = useMemo(() => {
    if (!geoJsonData?.features) return []
    
    return geoJsonData.features.map((feature: any, index: number) => {
      if (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon') {
        const coordinates = feature.geometry.type === 'Polygon' 
          ? [feature.geometry.coordinates[0]]
          : feature.geometry.coordinates.map((poly: any) => poly[0])
        
        return coordinates.map((ring: any, ringIndex: number) => {
          const points = ring.map((coord: [number, number]) => 
            latLngToVector3(coord[1], coord[0], EARTH_RADIUS + 0.01)
          )
          points.push(points[0]) // Close the polygon
          
          const geometry = new BufferGeometry().setFromPoints(points)
          return (
            <line key={`${index}-${ringIndex}`} geometry={geometry}>
              <lineBasicMaterial color="#ffffff" opacity={0.6} transparent />
            </line>
          )
        })
      }
      return null
    }).filter(Boolean).flat()
  }, [geoJsonData])

  return (
    <group ref={linesRef}>
      {boundaryLines}
    </group>
  )
}

// Main Globe3D component
const Globe3D: React.FC<Globe3DProps> = ({ onTileClick, selectedTile, analysisResults = [] }) => {
  const { eoTiles, geoJsonData } = useEODataStore()
  const [hoveredTile, setHoveredTile] = useState<EOTile | null>(null)

  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [0, 0, 15], fov: 60 }}
        gl={{ antialias: true, alpha: true }}
        onCreated={({ gl }) => {
          gl.setClearColor('#000011', 1)
        }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />

        {/* Earth */}
        <EarthSphere />

        {/* GeoJSON Boundaries */}
        {geoJsonData && <GeoJSONBoundaries geoJsonData={geoJsonData} />}

        {/* EO Tiles */}
        {eoTiles.map((tile) => (
          <EOTileMarker
            key={tile.id}
            tile={tile}
            onClick={() => onTileClick?.(tile)}
            isSelected={selectedTile?.id === tile.id}
          />
        ))}

        {/* Analysis Results */}
        {analysisResults.map((result) => (
          <AnalysisOverlay key={result.id} result={result} />
        ))}

        {/* Controls */}
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={8}
          maxDistance={50}
          maxPolarAngle={Math.PI}
        />

        {/* Stars background */}
        <mesh>
          <sphereGeometry args={[100, 32, 32]} />
          <meshBasicMaterial 
            color="#000033" 
            side={THREE.BackSide}
            transparent
            opacity={0.8}
          />
        </mesh>
      </Canvas>

      {/* UI Overlay */}
      {hoveredTile && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute top-4 left-4 bg-black/80 text-white p-4 rounded-lg backdrop-blur-sm"
        >
          <h3 className="font-semibold mb-2">EO Tile Information</h3>
          <p className="text-sm text-gray-300">
            Coordinates: {hoveredTile.latitude.toFixed(4)}, {hoveredTile.longitude.toFixed(4)}
          </p>
          <p className="text-sm text-gray-300">
            Captured: {new Date(hoveredTile.captureDate).toLocaleDateString()}
          </p>
          <p className="text-sm text-gray-300">
            Status: {hoveredTile.analysisStatus}
          </p>
        </motion.div>
      )}
    </div>
  )
}

export default Globe3D
