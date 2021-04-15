import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x4BE488EBc71Ac2D7a6b38aAa85ab0D9ff8130DBD';

export default new web3.eth.Contract(datasetDatabase, address);