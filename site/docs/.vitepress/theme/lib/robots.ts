/**
 * Robots available in the interactive scenes. Each one has `models/<id>/` exported by
 * `python -m rczoo.export.web` (meshes, robot.json, plant.xml, fixtures.json) and
 * `python -m rczoo.export.onnx_export` (reach.onnx).
 */
export interface RobotOption {
  id: 'panda' | 'sr3'
  /** Short label for selectors. */
  label: string
  /** One-line description shown under the selector. */
  description: string
}

export const ROBOTS: RobotOption[] = [
  {
    id: 'panda',
    label: 'Franka Panda',
    description: '7 自由度力矩控制协作臂（关节力矩传感器）：冗余，任务空间控制器在零空间里保持肘部姿态',
  },
  {
    id: 'sr3',
    label: '珞石 xMate SR3-C',
    description: '6 自由度协作臂（关节力矩传感器）：非冗余，零空间为空；末端装 60 mm 探针与六维力传感器',
  },
]

export type RobotId = RobotOption['id']

export const robotOption = (id: string): RobotOption => ROBOTS.find((r) => r.id === id) ?? ROBOTS[0]
