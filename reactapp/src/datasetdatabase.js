import web3 from './web3';
import { datasetDatabase} from "./abi/abi";

const address = '0xB35a93D015c254A7AA7B9C4A93C96Ac9d673ae2A';

export default new web3.eth.Contract(datasetDatabase, address);